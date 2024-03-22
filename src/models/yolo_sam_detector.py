"""
YOLO + SAM pipeline: two-stage detection + segmentation.

Stage 1 — YOLO (YOLOv8 or v11) detects bounding boxes around objects.
Stage 2 — SAM (Segment Anything Model) refines each detected box into
           a precise pixel-level segmentation mask.

This gives more accurate object boundaries than bounding-box detection alone —
useful for estimating trash volume or filtering out background bio/coral.

SAM model: vit_h (most accurate) or vit_b (faster)
SAM weights: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

Install:
    pip install 'git+https://github.com/facebookresearch/segment-anything.git'
    pip install supervision
"""

from pathlib import Path
from typing import Optional
import numpy as np


SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"

CLASS_NAMES = ['trash', 'bio', 'rov']
CLASS_COLORS = {
    'trash': (220, 0, 0),
    'bio':   (0, 180, 0),
    'rov':   (0, 130, 200),
}


def download_sam_weights(save_dir: str = 'weights') -> str:
    import urllib.request
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(save_dir) / 'sam_vit_h.pth'
    if not out_path.exists():
        print(f"Downloading SAM weights ({SAM_CHECKPOINT_URL})...")
        urllib.request.urlretrieve(SAM_CHECKPOINT_URL, out_path)
        print(f"Saved: {out_path}")
    return str(out_path)


class YOLOSAMPipeline:
    """
    Two-stage pipeline: YOLO detection → SAM segmentation.

    Usage:
        pipeline = YOLOSAMPipeline(
            yolo_weights='results/yolov8n/weights/best.pt',
            sam_checkpoint='weights/sam_vit_h.pth',
        )
        masks, boxes, labels = pipeline.run(image)
        annotated = pipeline.draw(image, masks, boxes, labels)
    """

    def __init__(self, yolo_weights: str, sam_checkpoint: str,
                 sam_model_type: str = SAM_MODEL_TYPE,
                 device: str = 'cuda'):
        import torch
        from ultralytics import YOLO
        from segment_anything import sam_model_registry, SamPredictor

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"YOLOSAMPipeline using: {self.device}")

        print(f"Loading YOLO weights: {yolo_weights}")
        self.yolo = YOLO(yolo_weights)

        print(f"Loading SAM ({sam_model_type}): {sam_checkpoint}")
        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)

    def run(self, image_bgr: np.ndarray, conf: float = 0.25, iou: float = 0.45):
        """
        Run YOLO detection then SAM segmentation on each detected box.

        Args:
            image_bgr: input image in BGR format (OpenCV)
            conf: YOLO confidence threshold
            iou: YOLO NMS IoU threshold

        Returns:
            masks: list of binary masks (H, W) for each detection
            boxes: list of [x1, y1, x2, y2]
            labels: list of class indices
            scores: list of confidence scores
        """
        import cv2

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Stage 1: YOLO detection
        yolo_results = self.yolo(image_bgr, conf=conf, iou=iou, verbose=False)[0]

        if len(yolo_results.boxes) == 0:
            return [], [], [], []

        boxes_xyxy = yolo_results.boxes.xyxy.cpu().numpy()
        labels     = yolo_results.boxes.cls.cpu().numpy().astype(int)
        scores     = yolo_results.boxes.conf.cpu().numpy()

        # Stage 2: SAM segmentation using YOLO boxes as prompts
        self.sam_predictor.set_image(image_rgb)

        all_masks = []
        for box in boxes_xyxy:
            masks, seg_scores, _ = self.sam_predictor.predict(
                box=box,
                multimask_output=True,
            )
            best_mask = masks[np.argmax(seg_scores)]  # take highest-confidence mask
            all_masks.append(best_mask)

        return all_masks, boxes_xyxy.tolist(), labels.tolist(), scores.tolist()

    def draw(self, image_bgr: np.ndarray, masks: list, boxes: list,
             labels: list, scores: list, alpha: float = 0.4) -> np.ndarray:
        """
        Draw segmentation masks and YOLO boxes onto the image.
        """
        import cv2
        annotated = image_bgr.copy()
        overlay = image_bgr.copy()

        for mask, box, label_idx, score in zip(masks, boxes, labels, scores):
            class_name = CLASS_NAMES[label_idx] if label_idx < len(CLASS_NAMES) else 'unknown'
            color_rgb = CLASS_COLORS.get(class_name, (128, 128, 128))
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

            # Fill segmentation mask
            overlay[mask > 0] = color_bgr

            # Draw bounding box
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 2)
            label_text = f"{class_name} {score:.0%}"
            cv2.putText(annotated, label_text, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 2, cv2.LINE_AA)

        # Blend mask overlay
        cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)
        return annotated

    @staticmethod
    def mask_area_pixels(mask: np.ndarray) -> int:
        """Return the number of pixels in a binary mask."""
        return int(np.sum(mask))

    @staticmethod
    def estimate_trash_coverage(masks: list, labels: list, image_shape: tuple) -> dict:
        """
        Estimate what fraction of the image is covered by trash vs bio vs background.
        """
        h, w = image_shape[:2]
        total_pixels = h * w

        coverage = {'trash': 0, 'bio': 0, 'rov': 0}
        for mask, label_idx in zip(masks, labels):
            class_name = CLASS_NAMES[label_idx] if label_idx < len(CLASS_NAMES) else None
            if class_name in coverage:
                coverage[class_name] += int(np.sum(mask))

        return {k: round(v / total_pixels * 100, 2) for k, v in coverage.items()}
