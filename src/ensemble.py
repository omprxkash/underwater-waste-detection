"""
Weighted ensemble inference across two trained models.

Running two models (YOLOv8n + YOLOv8m) and merging their predictions with
weighted box fusion gives better results than either model alone — especially
for small or partially occluded trash objects where model confidence varies.

Usage:
    python src/ensemble.py \
        --weights1 results/baseline/weights/yolov8n_best.pt \
        --weights2 results/baseline/weights/yolov8m_best.pt \
        --source path/to/image.jpg \
        --weight1 0.4 --weight2 0.6
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


CLASS_NAMES  = ['trash', 'bio', 'rov']
CLASS_COLORS = {'trash': (0, 0, 220), 'bio': (0, 180, 0), 'rov': (200, 130, 0)}


def parse_args():
    parser = argparse.ArgumentParser(description='Ensemble two YOLOv8 models for underwater detection')
    parser.add_argument('--weights1', type=str, required=True, help='Path to first model weights')
    parser.add_argument('--weights2', type=str, required=True, help='Path to second model weights')
    parser.add_argument('--source', type=str, required=True, help='Image or directory path')
    parser.add_argument('--weight1', type=float, default=0.4, help='Score weight for model 1 (0-1)')
    parser.add_argument('--weight2', type=float, default=0.6, help='Score weight for model 2 (0-1)')
    parser.add_argument('--conf', type=float, default=0.20, help='Confidence threshold')
    parser.add_argument('--iou_merge', type=float, default=0.5, help='IoU threshold for box merging')
    parser.add_argument('--output', type=str, default='results/ensemble', help='Output directory')
    return parser.parse_args()


def boxes_iou(box_a: list, box_b: list) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x_left   = max(box_a[0], box_b[0])
    y_top    = max(box_a[1], box_b[1])
    x_right  = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return intersection / (area_a + area_b - intersection + 1e-6)


def weighted_box_fusion(detections_a: list, detections_b: list,
                         weight_a: float, weight_b: float,
                         iou_threshold: float = 0.5) -> list:
    """
    Merge two sets of detections using weighted box fusion.

    Each detection: [x1, y1, x2, y2, confidence, class_id]
    Boxes from both models that overlap above iou_threshold are merged:
    - Final box = weighted average of coordinates
    - Final confidence = weighted average of scores
    """
    all_detections = [(d, weight_a) for d in detections_a] + [(d, weight_b) for d in detections_b]
    merged = []
    used = [False] * len(all_detections)

    for i, (det_i, w_i) in enumerate(all_detections):
        if used[i]:
            continue

        cluster = [(det_i, w_i)]
        used[i] = True

        for j, (det_j, w_j) in enumerate(all_detections):
            if used[j] or j == i:
                continue
            if det_i[5] != det_j[5]:  # different class
                continue
            if boxes_iou(det_i[:4], det_j[:4]) >= iou_threshold:
                cluster.append((det_j, w_j))
                used[j] = True

        # Weighted average box
        total_weight = sum(w for _, w in cluster)
        fused_box = [0.0] * 6
        for det, w in cluster:
            norm_w = w / total_weight
            for k in range(4):
                fused_box[k] += det[k] * norm_w
            fused_box[4] += det[4] * norm_w
        fused_box[5] = cluster[0][0][5]  # class id from first detection in cluster

        if fused_box[4] >= 0.0:  # confidence filter applied later
            merged.append(fused_box)

    return merged


def run_ensemble_on_image(model_a, model_b, image: np.ndarray,
                           weight_a: float, weight_b: float,
                           conf: float, iou_merge: float) -> list:
    """Run both models and return merged detections."""
    def extract_detections(results):
        dets = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            cls_id = int(box.cls[0])
            if confidence >= conf:
                dets.append([x1, y1, x2, y2, confidence, cls_id])
        return dets

    dets_a = extract_detections(model_a(image, verbose=False))
    dets_b = extract_detections(model_b(image, verbose=False))

    return weighted_box_fusion(dets_a, dets_b, weight_a, weight_b, iou_merge)


def draw_fused_detections(image: np.ndarray, detections: list) -> np.ndarray:
    annotated = image.copy()
    for det in detections:
        x1, y1, x2, y2, conf_score, cls_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else 'unknown'
        color = CLASS_COLORS.get(label, (128, 128, 128))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf_score:.0%}"
        cv2.putText(annotated, text, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    n_trash = sum(1 for d in detections if d[5] == 0)
    cv2.putText(annotated, f"Ensemble | total={len(detections)} trash={n_trash}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return annotated


def run_ensemble(args) -> None:
    print(f"Loading model 1: {args.weights1}  (weight={args.weight1})")
    model_a = YOLO(args.weights1)

    print(f"Loading model 2: {args.weights2}  (weight={args.weight2})")
    model_b = YOLO(args.weights2)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = Path(args.source)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    if source.is_file() and source.suffix.lower() in image_extensions:
        image_files = [source]
    elif source.is_dir():
        image_files = [f for f in source.rglob('*') if f.suffix.lower() in image_extensions]
    else:
        print(f"Source not found or unsupported: {source}")
        return

    print(f"Running ensemble on {len(image_files)} image(s)...")
    for img_path in image_files:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        merged_dets = run_ensemble_on_image(
            model_a, model_b, frame,
            args.weight1, args.weight2,
            args.conf, args.iou_merge
        )

        annotated = draw_fused_detections(frame, merged_dets)
        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), annotated)

        n_trash = sum(1 for d in merged_dets if d[5] == 0)
        print(f"  {img_path.name}: {len(merged_dets)} detections ({n_trash} trash)")

    print(f"\nEnsemble results saved to: {output_dir}")


if __name__ == '__main__':
    args = parse_args()
    run_ensemble(args)
