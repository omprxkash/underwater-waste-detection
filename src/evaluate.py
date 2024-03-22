"""
Evaluation utilities — compute metrics, plot confusion matrix, visualize predictions.

Usage:
    python src/evaluate.py --weights results/baseline/weights/best.pt --data configs/dataset.yaml
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import cv2
from ultralytics import YOLO


CLASS_NAMES = ['trash', 'bio', 'rov']
CLASS_COLORS = {
    'trash': (0, 0, 220),    # red (BGR)
    'bio':   (0, 180, 0),    # green
    'rov':   (200, 130, 0),  # blue-ish
}


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained underwater detection model')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to best.pt weights file')
    parser.add_argument('--data', type=str, default='configs/dataset.yaml',
                        help='Dataset YAML config')
    parser.add_argument('--output', type=str, default='results/baseline',
                        help='Directory to save evaluation outputs')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='NMS IoU threshold')
    parser.add_argument('--split', type=str, default='test',
                        choices=['val', 'test'],
                        help='Which split to evaluate on')
    return parser.parse_args()


def run_validation(weights_path: str, data_config: str, split: str = 'test',
                   conf: float = 0.25, iou: float = 0.45) -> dict:
    detector = YOLO(weights_path)
    val_results = detector.val(
        data=data_config,
        split=split,
        conf=conf,
        iou=iou,
        verbose=True,
    )
    return val_results


def plot_confusion_matrix(confusion_data: np.ndarray, class_names: list, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        confusion_data,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names + ['background'],
        yticklabels=class_names + ['background'],
        ax=ax,
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Ground Truth', fontsize=12)
    ax.set_title('Confusion Matrix — Underwater Waste Detection', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved: {output_path}")


def draw_detections(image: np.ndarray, boxes, class_names: list) -> np.ndarray:
    annotated = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_idx = int(box.cls[0].item())
        conf_score = float(box.conf[0].item())
        label = class_names[cls_idx] if cls_idx < len(class_names) else 'unknown'
        color = CLASS_COLORS.get(label, (128, 128, 128))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf_score:.2f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(annotated, (x1, y1 - text_size[1] - 4), (x1 + text_size[0] + 4, y1), color, -1)
        cv2.putText(annotated, text, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return annotated


def visualize_sample_predictions(weights_path: str, image_paths: list,
                                  class_names: list, output_dir: str,
                                  conf: float = 0.25, n_samples: int = 8) -> None:
    detector = YOLO(weights_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sampled = image_paths[:n_samples]
    cols = 4
    rows = (len(sampled) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten() if rows > 1 else [axes] if len(sampled) == 1 else axes.flatten()

    for idx, img_path in enumerate(sampled):
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        detection_results = detector(image, conf=conf, verbose=False)[0]
        annotated_image = draw_detections(image, detection_results.boxes, class_names)
        annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        axes[idx].imshow(annotated_rgb)
        axes[idx].axis('off')
        axes[idx].set_title(Path(img_path).stem[:20], fontsize=8)

    for idx in range(len(sampled), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Sample Predictions — Underwater Waste Detection', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_path = output_path / 'sample_predictions.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sample predictions saved: {save_path}")


def save_metrics_summary(val_results, output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics = {
        'mAP50': float(val_results.box.map50),
        'mAP50_95': float(val_results.box.map),
        'precision': float(val_results.box.mp),
        'recall': float(val_results.box.mr),
        'per_class_mAP50': val_results.box.ap50.tolist(),
        'class_names': CLASS_NAMES,
    }

    with open(output_path / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*50}")
    print("EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"  mAP@0.5:       {metrics['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95:  {metrics['mAP50_95']:.4f}")
    print(f"  Precision:     {metrics['precision']:.4f}")
    print(f"  Recall:        {metrics['recall']:.4f}")
    print(f"\nPer-class mAP@0.5:")
    for cls_name, ap in zip(CLASS_NAMES, metrics['per_class_mAP50']):
        print(f"  {cls_name:<10} {ap:.4f}")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    args = parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)

    print(f"Running evaluation on {args.split} split...")
    val_results = run_validation(args.weights, args.data, args.split, args.conf, args.iou)
    save_metrics_summary(val_results, args.output)
