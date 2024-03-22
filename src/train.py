"""
Training script — YOLOv8 on underwater trash detection dataset.
Wraps the ultralytics training API with our config files.

Usage:
    python src/train.py --model yolov8n --epochs 100 --batch 16
    python src/train.py --model yolov8m --epochs 100 --batch 8 --resume
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOv8 for underwater waste detection')
    parser.add_argument('--model', type=str, default='yolov8n',
                        choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
                        help='YOLOv8 variant to train')
    parser.add_argument('--data', type=str, default='configs/dataset.yaml',
                        help='Path to dataset YAML config')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size (reduce to 8 if GPU OOM)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Training image size')
    parser.add_argument('--project', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--name', type=str, default='baseline',
                        help='Run name (subfolder in project dir)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')
    parser.add_argument('--device', type=str, default='0',
                        help='Device: 0 for GPU, cpu for CPU')
    return parser.parse_args()


def load_hyperparams(config_path: str = 'configs/hyperparams.yaml') -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_training(args):
    model_path = f"{args.model}.pt"
    print(f"\nLoading model: {model_path}")
    detector = YOLO(model_path)

    hp = load_hyperparams()

    print(f"Starting training — {args.epochs} epochs, batch={args.batch}, imgsz={args.imgsz}")
    print(f"Dataset config: {args.data}")
    print(f"Output: {args.project}/{args.name}\n")

    training_results = detector.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
        optimizer=hp.get('optimizer', 'AdamW'),
        lr0=hp.get('lr0', 0.001),
        lrf=hp.get('lrf', 0.01),
        momentum=hp.get('momentum', 0.937),
        weight_decay=hp.get('weight_decay', 0.0005),
        warmup_epochs=hp.get('warmup_epochs', 3),
        mosaic=hp.get('mosaic', 1.0),
        mixup=hp.get('mixup', 0.1),
        save_period=hp.get('save_period', 10),
        verbose=True,
    )

    print(f"\nTraining complete. Best weights saved to: {args.project}/{args.name}/weights/best.pt")
    return training_results


if __name__ == '__main__':
    args = parse_args()
    run_training(args)
