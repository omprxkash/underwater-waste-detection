"""
YOLO-NAS wrapper for underwater waste detection.
Uses the super-gradients library (Deci AI).

Install: pip install super-gradients==3.7.1

YOLO-NAS (Neural Architecture Search) variants:
  - yolo_nas_s: small — fastest, lowest accuracy
  - yolo_nas_m: medium — balanced
  - yolo_nas_l: large — highest accuracy, most compute

The NAS search optimizes accuracy/latency specifically for edge deployment,
making it relevant for AUV compute constraints.
"""

from pathlib import Path
from typing import Optional


CLASS_NAMES = ['trash', 'bio', 'rov']

YOLO_NAS_VARIANTS = ['yolo_nas_s', 'yolo_nas_m', 'yolo_nas_l']


def check_supergradients():
    try:
        import super_gradients
        return True
    except ImportError:
        print("super-gradients not installed. Run: pip install super-gradients==3.7.1")
        return False


class YOLONASDetector:
    """
    YOLO-NAS detector wrapper using super-gradients Trainer API.
    """

    def __init__(self, variant: str = 'yolo_nas_s', num_classes: int = 3):
        if not check_supergradients():
            raise ImportError("Install super-gradients: pip install super-gradients==3.7.1")

        if variant not in YOLO_NAS_VARIANTS:
            raise ValueError(f"Choose from: {YOLO_NAS_VARIANTS}")

        self.variant = variant
        self.num_classes = num_classes
        self.model = None

    def _load_model(self):
        from super_gradients.training import models
        self.model = models.get(self.variant, num_classes=self.num_classes)

    def build_dataloaders(self, dataset_dir: str, batch_size: int = 8):
        from super_gradients.training.dataloaders.dataloaders import (
            coco_detection_yolo_format_train,
            coco_detection_yolo_format_val,
        )

        shared_params = {
            'data_dir': dataset_dir,
            'classes': CLASS_NAMES,
        }

        train_loader = coco_detection_yolo_format_train(
            dataset_params={**shared_params,
                            'images_dir': f'{dataset_dir}/images/train',
                            'labels_dir': f'{dataset_dir}/labels/train'},
            dataloader_params={'batch_size': batch_size, 'num_workers': 2},
        )

        val_loader = coco_detection_yolo_format_val(
            dataset_params={**shared_params,
                            'images_dir': f'{dataset_dir}/images/val',
                            'labels_dir': f'{dataset_dir}/labels/val'},
            dataloader_params={'batch_size': batch_size, 'num_workers': 2},
        )

        test_loader = coco_detection_yolo_format_val(
            dataset_params={**shared_params,
                            'images_dir': f'{dataset_dir}/images/test',
                            'labels_dir': f'{dataset_dir}/labels/test'},
            dataloader_params={'batch_size': batch_size, 'num_workers': 2},
        )

        return train_loader, val_loader, test_loader

    def build_train_params(self, max_epochs: int = 100, initial_lr: float = 5e-4) -> dict:
        from super_gradients.training.losses import PPYoloELoss
        from super_gradients.training.metrics import DetectionMetrics_050
        from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback

        return {
            'silent_mode': False,
            'average_best_models': True,
            'warmup_mode': 'linear_epoch_step',
            'warmup_initial_lr': 1e-6,
            'lr_warmup_epochs': 3,
            'initial_lr': initial_lr,
            'lr_mode': 'cosine',
            'cosine_final_lr_ratio': 0.1,
            'optimizer': 'Adam',
            'optimizer_params': {'weight_decay': 0.0001},
            'zero_weight_decay_on_bias_and_bn': True,
            'ema': True,
            'ema_params': {'decay': 0.9, 'decay_type': 'threshold'},
            'max_epochs': max_epochs,
            'mixed_precision': True,
            'loss': PPYoloELoss(
                use_static_assigner=False,
                num_classes=self.num_classes,
                reg_max=16,
            ),
            'valid_metrics_list': [
                DetectionMetrics_050(
                    score_thres=0.1,
                    top_k_predictions=300,
                    num_cls=self.num_classes,
                    normalize_targets=True,
                    post_prediction_callback=PPYoloEPostPredictionCallback(
                        score_threshold=0.01,
                        nms_top_k=1000,
                        max_predictions=300,
                        nms_threshold=0.7,
                    ),
                )
            ],
            'metric_to_watch': 'mAP@0.50',
        }

    def train(self, dataset_dir: str, checkpoint_dir: str,
              experiment_name: str = 'yolo_nas_underwater',
              max_epochs: int = 100, batch_size: int = 8) -> None:
        from super_gradients.training import Trainer

        self._load_model()
        trainer = Trainer(experiment_name=experiment_name, ckpt_root_dir=checkpoint_dir)
        train_loader, val_loader, _ = self.build_dataloaders(dataset_dir, batch_size)
        train_params = self.build_train_params(max_epochs)

        trainer.train(
            model=self.model,
            training_params=train_params,
            train_loader=train_loader,
            valid_loader=val_loader,
        )

    def load_weights(self, checkpoint_path: str, num_classes: Optional[int] = None) -> None:
        from super_gradients.training import models
        nc = num_classes or self.num_classes
        self.model = models.get(
            self.variant,
            num_classes=nc,
            checkpoint_path=checkpoint_path,
        )

    def predict(self, image_path: str, conf: float = 0.4):
        if self.model is None:
            raise RuntimeError("Load weights first via load_weights()")
        return self.model.predict(image_path, conf=conf)
