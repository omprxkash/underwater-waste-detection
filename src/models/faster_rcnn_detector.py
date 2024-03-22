"""
Faster R-CNN detector for underwater waste detection.
Uses torchvision's ResNet-50 FPN backbone with custom head.

Why include Faster R-CNN?
- Strong anchor-based two-stage detector — good baseline to compare against single-stage YOLO
- ResNet-50 FPN provides multi-scale feature extraction, useful for varied trash sizes
- Well-studied training dynamics make it a stable academic comparison point
"""

import copy
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


CLASS_NAMES = ['background', 'trash', 'bio', 'rov']
NUM_CLASSES = len(CLASS_NAMES)


class TrashCanDetectionDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for TrashCan annotations loaded from a pre-processed pickle file.
    Annotations pickled from TrashCan JSON with columns:
        image_id, file_name, bbox (COCO [x,y,w,h]), category_id
    """

    def __init__(self, pkl_path: str, images_dir: str, transforms=None):
        self.data = pd.read_pickle(pkl_path)
        self.images_dir = Path(images_dir)
        self.transforms = transforms
        self.image_ids = sorted(self.data['image_id'].unique())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]
        rows = self.data[self.data['image_id'] == img_id]

        img_path = self.images_dir / rows['file_name'].iloc[0]
        image = Image.open(img_path).convert('RGB')

        # Convert COCO [x, y, w, h] → [x1, y1, x2, y2]
        raw_boxes = copy.deepcopy(list(rows['bbox']))
        boxes = []
        for b in raw_boxes:
            x1, y1 = b[0], b[1]
            x2, y2 = b[0] + b[2], b[1] + b[3]
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        labels = torch.as_tensor(rows['category_id'].values, dtype=torch.int64)
        area = (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * (boxes_tensor[:, 2] - boxes_tensor[:, 0])

        target = {
            'boxes':    boxes_tensor,
            'labels':   labels[:len(boxes)],
            'image_id': torch.tensor([img_id]),
            'area':     area,
            'iscrowd':  torch.zeros(len(boxes), dtype=torch.int64),
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_faster_rcnn(num_classes: int = NUM_CLASSES, pretrained_backbone: bool = True):
    """
    Load a Faster R-CNN with ResNet-50 FPN backbone and replace the classification head.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights='DEFAULT' if pretrained_backbone else None
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch: int, print_freq: int = 20):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        n_batches += 1

        if batch_idx % print_freq == 0:
            avg = total_loss / n_batches
            print(f"  Epoch [{epoch}] batch [{batch_idx}/{len(data_loader)}]  loss={losses.item():.4f}  avg={avg:.4f}")

    return total_loss / max(n_batches, 1)


def run_training(pkl_train: str, pkl_val: str, images_train_dir: str, images_val_dir: str,
                 save_path: str, num_classes: int = NUM_CLASSES,
                 num_epochs: int = 10, batch_size: int = 4, lr: float = 0.002) -> None:
    import torchvision.transforms as T

    transform = T.Compose([T.ToTensor()])

    train_set = TrashCanDetectionDataset(pkl_train, images_train_dir, transforms=transform)
    val_set   = TrashCanDetectionDataset(pkl_val, images_val_dir, transforms=transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=2, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_fn
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Training Faster R-CNN on {device}")

    model = build_faster_rcnn(num_classes=num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    best_loss = float('inf')
    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        lr_scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs} — avg train loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model → {save_path}")

    print(f"Training complete. Best weights: {save_path}")


def run_inference(model_path: str, image: np.ndarray,
                  num_classes: int = NUM_CLASSES, conf_threshold: float = 0.5):
    """
    Run Faster R-CNN inference on a single numpy image (H, W, 3) BGR.
    Returns list of (box, score, label_idx).
    """
    import torchvision.transforms as T

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_faster_rcnn(num_classes=num_classes, pretrained_backbone=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    pil_image = Image.fromarray(image[:, :, ::-1])  # BGR→RGB
    tensor = T.ToTensor()(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(tensor)

    pred = predictions[0]
    results = []
    for box, score, label in zip(pred['boxes'], pred['scores'], pred['labels']):
        if score.item() >= conf_threshold:
            results.append((box.cpu().numpy().tolist(), float(score), int(label)))

    return results
