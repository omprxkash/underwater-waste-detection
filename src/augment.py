"""
Custom underwater-specific data augmentation.

Standard augmentations (flip, crop, color jitter) don't model the physics of
underwater imagery. These simulate real underwater degradation effects to make
the model more robust to domain variation.
"""

import cv2
import numpy as np
import random
from typing import Tuple


def add_synthetic_haze(image: np.ndarray, intensity: float = 0.4) -> np.ndarray:
    """
    Simulate underwater haze/turbidity by blending the image with a
    blue-green tinted fog layer. Mimics backscatter from suspended particles.
    """
    haze_color = np.array([180, 140, 80], dtype=np.float32)  # BGR: blue-green haze
    haze_layer = np.ones_like(image, dtype=np.float32) * haze_color

    alpha = random.uniform(0.1, intensity)
    hazed = cv2.addWeighted(image.astype(np.float32), 1 - alpha, haze_layer, alpha, 0)
    return np.clip(hazed, 0, 255).astype(np.uint8)


def apply_color_cast(image: np.ndarray, cast_strength: float = 0.3) -> np.ndarray:
    """
    Apply a random blue or green color cast to simulate different water column depths.
    Shallow water → green cast. Deep water → blue cast.
    """
    image_float = image.astype(np.float32)
    cast_type = random.choice(['blue', 'green'])

    strength = random.uniform(0.0, cast_strength)

    if cast_type == 'blue':
        image_float[:, :, 0] = np.clip(image_float[:, :, 0] * (1 + strength), 0, 255)  # boost B
        image_float[:, :, 2] = np.clip(image_float[:, :, 2] * (1 - strength * 0.5), 0, 255)  # suppress R
    else:
        image_float[:, :, 1] = np.clip(image_float[:, :, 1] * (1 + strength), 0, 255)  # boost G
        image_float[:, :, 2] = np.clip(image_float[:, :, 2] * (1 - strength * 0.5), 0, 255)  # suppress R

    return image_float.astype(np.uint8)


def apply_motion_blur(image: np.ndarray, max_kernel: int = 9) -> np.ndarray:
    """
    Apply directional motion blur to simulate camera or object movement underwater.
    """
    kernel_size = random.choice([3, 5, 7, max_kernel])
    angle = random.uniform(0, 180)

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0 / kernel_size

    rotation_matrix = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
    rotated_kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))
    rotated_kernel /= rotated_kernel.sum() + 1e-6

    return cv2.filter2D(image, -1, rotated_kernel)


def add_caustic_pattern(image: np.ndarray, intensity: float = 0.15) -> np.ndarray:
    """
    Add a caustic light pattern (rippling light rays from surface refraction).
    These bright wavy patches are a distinctive feature of shallow underwater scenes.
    """
    h, w = image.shape[:2]

    x = np.linspace(0, 4 * np.pi, w)
    y = np.linspace(0, 4 * np.pi, h)
    xx, yy = np.meshgrid(x, y)

    caustic = np.sin(xx + random.uniform(0, np.pi)) * np.cos(yy + random.uniform(0, np.pi))
    caustic = (caustic - caustic.min()) / (caustic.max() - caustic.min() + 1e-6)
    caustic = (caustic * 255 * intensity).astype(np.uint8)
    caustic_3ch = np.stack([caustic, caustic, caustic], axis=-1)

    return np.clip(image.astype(np.int32) + caustic_3ch, 0, 255).astype(np.uint8)


def underwater_augment(
    image: np.ndarray,
    p_haze: float = 0.3,
    p_color_cast: float = 0.4,
    p_motion_blur: float = 0.2,
    p_caustic: float = 0.2,
) -> np.ndarray:
    """
    Apply a random combination of underwater-specific augmentations.

    Args:
        image: BGR image as numpy array
        p_*: probability of applying each augmentation

    Returns:
        Augmented BGR image
    """
    augmented = image.copy()

    if random.random() < p_haze:
        augmented = add_synthetic_haze(augmented)
    if random.random() < p_color_cast:
        augmented = apply_color_cast(augmented)
    if random.random() < p_motion_blur:
        augmented = apply_motion_blur(augmented)
    if random.random() < p_caustic:
        augmented = add_caustic_pattern(augmented)

    return augmented


if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 2:
        img = cv2.imread(sys.argv[1])
        if img is not None:
            result = underwater_augment(img)
            out_path = sys.argv[2] if len(sys.argv) >= 3 else 'augmented_output.jpg'
            cv2.imwrite(out_path, result)
            print(f"Saved augmented image to {out_path}")
