"""
YOLOv8 / YOLOv11 wrapper for underwater waste detection.
Supports training, evaluation, and inference for nano/medium/extra-large variants.
"""

from pathlib import Path
from ultralytics import YOLO


SUPPORTED_VARIANTS = {
    # YOLOv8 family
    'yolov8n': 'yolov8n.pt',
    'yolov8m': 'yolov8m.pt',
    'yolov8x': 'yolov8x.pt',
    # YOLOv11 family (newest)
    'yolo11n': 'yolo11n.pt',
    'yolo11m': 'yolo11m.pt',
    'yolo11x': 'yolo11x.pt',
}

CLASS_NAMES = ['trash', 'bio', 'rov']


class YOLODetector:
    """
    Thin wrapper around Ultralytics YOLO for underwater waste detection.
    Handles train / validate / predict with consistent interface.
    """

    def __init__(self, variant: str = 'yolov8n', weights_path: str = None):
        """
        Args:
            variant: model variant key from SUPPORTED_VARIANTS
            weights_path: path to fine-tuned weights (overrides variant default)
        """
        if weights_path and Path(weights_path).exists():
            self.model = YOLO(weights_path)
            self.variant = variant
        elif variant in SUPPORTED_VARIANTS:
            self.model = YOLO(SUPPORTED_VARIANTS[variant])
            self.variant = variant
        else:
            raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(SUPPORTED_VARIANTS)}")

    def train(self, data_yaml: str, output_dir: str, run_name: str = None,
              epochs: int = 100, batch: int = 16, imgsz: int = 640,
              device: str = '0', **kwargs) -> dict:
        """
        Train the detector.
        Returns dict with best_model_path and training results.
        """
        name = run_name or f"{self.variant}_baseline"
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=output_dir,
            name=name,
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            warmup_epochs=3,
            mosaic=1.0,
            mixup=0.1,
            verbose=True,
            **kwargs,
        )
        best_weights = Path(output_dir) / name / 'weights' / 'best.pt'
        return {'results': results, 'best_weights': str(best_weights), 'run_name': name}

    def evaluate(self, data_yaml: str, split: str = 'val',
                 conf: float = 0.25, iou: float = 0.45) -> dict:
        """
        Run validation and return metrics dict.
        """
        val_results = self.model.val(data=data_yaml, split=split, conf=conf, iou=iou)
        return {
            'model': self.variant,
            'mAP50': round(float(val_results.box.map50), 4),
            'mAP50_95': round(float(val_results.box.map), 4),
            'precision': round(float(val_results.box.mp), 4),
            'recall': round(float(val_results.box.mr), 4),
            'per_class_AP50': [round(float(v), 4) for v in val_results.box.ap50.tolist()],
            'class_names': CLASS_NAMES,
        }

    def predict(self, source, conf: float = 0.25, iou: float = 0.45,
                imgsz: int = 640, verbose: bool = False):
        """
        Run inference. Source can be image path, directory, or numpy array.
        Returns ultralytics Results list.
        """
        return self.model(source, conf=conf, iou=iou, imgsz=imgsz, verbose=verbose)

    def export(self, format: str = 'onnx', imgsz: int = 640) -> str:
        """Export model to ONNX, TensorRT, etc."""
        return self.model.export(format=format, imgsz=imgsz)
