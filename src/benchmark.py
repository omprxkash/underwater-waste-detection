"""
Unified benchmark runner — evaluate all trained models on the same test set
and produce a comparison table.

Usage:
    python src/benchmark.py \
        --data configs/dataset.yaml \
        --weights_dir results/ \
        --output results/benchmark_table.csv
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO


CLASS_NAMES = ['trash', 'bio', 'rov']


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark all trained models and produce comparison table')
    parser.add_argument('--data', type=str, default='configs/dataset.yaml',
                        help='YOLO dataset YAML config')
    parser.add_argument('--weights_dir', type=str, default='results/',
                        help='Directory containing subdirectories with best.pt files')
    parser.add_argument('--output', type=str, default='results/benchmark_table.csv',
                        help='Output CSV path for comparison table')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold for evaluation')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--speed_images', type=int, default=100,
                        help='Number of images for speed benchmark')
    return parser.parse_args()


def find_model_weights(weights_dir: str) -> dict:
    """
    Search weights_dir recursively for best.pt files.
    Returns dict: {run_name: path_to_best.pt}
    """
    found = {}
    weights_path = Path(weights_dir)

    for pt_file in weights_path.rglob('best.pt'):
        run_name = pt_file.parent.parent.name  # e.g. results/yolov8n_baseline/weights/best.pt → yolov8n_baseline
        found[run_name] = str(pt_file)

    return found


def benchmark_yolo_model(weights_path: str, data_yaml: str, run_name: str,
                          conf: float, iou: float, n_speed_images: int = 100) -> dict:
    """
    Evaluate a single YOLO model on val/test split and measure inference speed.
    Returns metrics dict.
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {run_name}")
    print(f"  Weights: {weights_path}")
    print(f"{'='*60}")

    detector = YOLO(weights_path)

    # Accuracy metrics
    val_results = detector.val(data=data_yaml, conf=conf, iou=iou, verbose=False)

    # Speed benchmark
    import yaml
    with open(data_yaml) as f:
        dataset_info = yaml.safe_load(f)

    dataset_root = Path(dataset_info.get('path', '.'))
    val_images_dir = dataset_root / dataset_info.get('val', 'images/val')
    if 'images' not in str(val_images_dir):
        val_images_dir = dataset_root / 'images' / dataset_info.get('val', 'val')

    image_files = list(val_images_dir.glob('*.jpg')) + list(val_images_dir.glob('*.png'))
    speed_sample = image_files[:n_speed_images]

    # Warmup
    if speed_sample:
        sample_img = cv2.imread(str(speed_sample[0]))
        for _ in range(5):
            detector(sample_img, verbose=False)

    # Timed run
    start = time.time()
    for img_path in speed_sample:
        frame = cv2.imread(str(img_path))
        if frame is not None:
            detector(frame, conf=conf, iou=iou, verbose=False)
    elapsed = time.time() - start

    fps = len(speed_sample) / elapsed if elapsed > 0 else 0
    ms_per_frame = (elapsed / len(speed_sample) * 1000) if speed_sample else 0

    per_class_ap50 = [round(float(v), 4) for v in val_results.box.ap50.tolist()]

    metrics = {
        'model': run_name,
        'mAP50': round(float(val_results.box.map50), 4),
        'mAP50_95': round(float(val_results.box.map), 4),
        'precision': round(float(val_results.box.mp), 4),
        'recall': round(float(val_results.box.mr), 4),
        'fps': round(fps, 1),
        'ms_per_frame': round(ms_per_frame, 1),
        'trash_AP50': per_class_ap50[0] if len(per_class_ap50) > 0 else None,
        'bio_AP50':   per_class_ap50[1] if len(per_class_ap50) > 1 else None,
        'rov_AP50':   per_class_ap50[2] if len(per_class_ap50) > 2 else None,
        'weights_path': weights_path,
    }

    print(f"  mAP@0.5:      {metrics['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95: {metrics['mAP50_95']:.4f}")
    print(f"  Precision:    {metrics['precision']:.4f}")
    print(f"  Recall:       {metrics['recall']:.4f}")
    print(f"  Speed:        {fps:.1f} FPS ({ms_per_frame:.1f} ms/frame)")

    return metrics


def plot_benchmark_comparison(df: pd.DataFrame, output_dir: str) -> None:
    """
    Generate comparison bar charts for all benchmarked models.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    palette = sns.color_palette('husl', len(df))

    metrics_to_plot = [
        ('mAP50', 'mAP @ 0.5', axes[0, 0]),
        ('mAP50_95', 'mAP @ 0.5:0.95', axes[0, 1]),
        ('precision', 'Precision', axes[1, 0]),
        ('fps', 'Inference Speed (FPS)', axes[1, 1]),
    ]

    for col, title, ax in metrics_to_plot:
        if col not in df.columns:
            continue
        bars = ax.bar(df['model'], df[col], color=palette, edgecolor='white', linewidth=1)
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_ylabel(col)
        ax.set_xticklabels(df['model'], rotation=30, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, df[col]):
            if pd.notna(val):
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                        f'{val:.3f}' if col != 'fps' else f'{val:.0f}',
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.suptitle('Model Benchmark — Underwater Waste Detection', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plot_path = output_path / 'benchmark_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nBenchmark chart saved: {plot_path}")


def run_benchmark(args) -> pd.DataFrame:
    model_weights = find_model_weights(args.weights_dir)

    if not model_weights:
        print(f"No best.pt files found in: {args.weights_dir}")
        print("Train at least one model first (notebooks/02_yolov8_training.ipynb)")
        return pd.DataFrame()

    print(f"Found {len(model_weights)} trained model(s):")
    for name, path in model_weights.items():
        print(f"  {name}: {path}")

    all_metrics = []
    for run_name, weights_path in sorted(model_weights.items()):
        try:
            metrics = benchmark_yolo_model(
                weights_path, args.data, run_name,
                args.conf, args.iou, args.speed_images,
            )
            all_metrics.append(metrics)
        except Exception as e:
            print(f"  Error benchmarking {run_name}: {e}")

    if not all_metrics:
        print("No models benchmarked successfully.")
        return pd.DataFrame()

    df = pd.DataFrame(all_metrics)
    df = df.sort_values('mAP50', ascending=False).reset_index(drop=True)

    # Save CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nBenchmark table saved: {output_path}")

    # Save JSON
    json_path = output_path.with_suffix('.json')
    df.to_json(json_path, orient='records', indent=2)

    # Print table
    display_cols = ['model', 'mAP50', 'mAP50_95', 'precision', 'recall', 'fps', 'trash_AP50', 'bio_AP50', 'rov_AP50']
    display_df = df[[c for c in display_cols if c in df.columns]]
    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS")
    print('='*80)
    print(display_df.to_string(index=False))
    print('='*80)

    # Plot
    plot_benchmark_comparison(df, str(output_path.parent))

    return df


if __name__ == '__main__':
    args = parse_args()
    run_benchmark(args)
