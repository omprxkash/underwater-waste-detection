"""
Run detection inference on a single image, a directory, or a video file.

Usage:
    # Single image
    python src/inference.py --weights results/baseline/weights/best.pt --source path/to/image.jpg

    # Directory of images
    python src/inference.py --weights results/baseline/weights/best.pt --source path/to/images/

    # Video file
    python src/inference.py --weights results/baseline/weights/best.pt --source path/to/video.mp4

    # With preprocessing enabled
    python src/inference.py --weights results/baseline/weights/best.pt --source image.jpg --enhance
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

from preprocess import enhance_underwater


CLASS_NAMES = ['trash', 'bio', 'rov']
CLASS_COLORS = {
    'trash': (0, 0, 220),
    'bio':   (0, 180, 0),
    'rov':   (200, 130, 0),
}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


def parse_args():
    parser = argparse.ArgumentParser(description='Run underwater waste detection on images or video')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to trained weights (best.pt)')
    parser.add_argument('--source', type=str, required=True,
                        help='Image file, directory, or video file')
    parser.add_argument('--output', type=str, default='results/inference',
                        help='Output directory for annotated results')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Detection confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='NMS IoU threshold')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Inference image size')
    parser.add_argument('--enhance', action='store_true',
                        help='Apply underwater image enhancement before detection')
    parser.add_argument('--save', action='store_true', default=True,
                        help='Save annotated output images')
    return parser.parse_args()


def annotate_frame(frame: np.ndarray, detection_results) -> np.ndarray:
    annotated = frame.copy()
    for box in detection_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_idx = int(box.cls[0].item())
        conf_score = float(box.conf[0].item())
        label = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else 'unknown'
        color = CLASS_COLORS.get(label, (128, 128, 128))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf_score:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    detection_count = len(detection_results.boxes)
    trash_count = sum(1 for b in detection_results.boxes if int(b.cls[0]) == 0)
    cv2.putText(annotated, f"Total: {detection_count} | Trash: {trash_count}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return annotated


def process_image(detector, image_path: Path, output_dir: Path,
                  conf: float, iou: float, imgsz: int, enhance: bool) -> None:
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"  Warning: could not read {image_path}")
        return

    input_frame = enhance_underwater(frame) if enhance else frame
    detection_results = detector(input_frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
    annotated = annotate_frame(frame, detection_results)

    out_path = output_dir / image_path.name
    cv2.imwrite(str(out_path), annotated)
    n_trash = sum(1 for b in detection_results.boxes if int(b.cls[0]) == 0)
    print(f"  {image_path.name}: {len(detection_results.boxes)} detections ({n_trash} trash)")


def process_video(detector, video_path: Path, output_dir: Path,
                  conf: float, iou: float, imgsz: int, enhance: bool) -> None:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = output_dir / f"{video_path.stem}_detected.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    print(f"Processing video: {video_path.name} ({total_frames} frames)")
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        input_frame = enhance_underwater(frame) if enhance else frame
        detection_results = detector(input_frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
        annotated = annotate_frame(frame, detection_results)
        writer.write(annotated)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames...")

    cap.release()
    writer.release()
    print(f"Video saved: {out_path}")


def run_inference(args) -> None:
    detector = YOLO(args.weights)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = Path(args.source)

    if source.is_file():
        ext = source.suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            print(f"Running on image: {source}")
            process_image(detector, source, output_dir, args.conf, args.iou, args.imgsz, args.enhance)
        elif ext in VIDEO_EXTENSIONS:
            print(f"Running on video: {source}")
            process_video(detector, source, output_dir, args.conf, args.iou, args.imgsz, args.enhance)
        else:
            print(f"Unsupported file type: {ext}")

    elif source.is_dir():
        image_files = [f for f in source.rglob('*') if f.suffix.lower() in IMAGE_EXTENSIONS]
        print(f"Found {len(image_files)} images in {source}")
        for img_path in image_files:
            process_image(detector, img_path, output_dir, args.conf, args.iou, args.imgsz, args.enhance)
    else:
        print(f"Source not found: {source}")

    print(f"\nDone. Outputs saved to: {output_dir}")


if __name__ == '__main__':
    args = parse_args()
    run_inference(args)
