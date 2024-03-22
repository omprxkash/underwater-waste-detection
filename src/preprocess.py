"""
Underwater image enhancement utilities.

Underwater images suffer from: color shift (blue-green cast), low contrast,
haze/turbidity, light absorption, and backscatter. These functions correct
for those effects before feeding images to the detector.
"""

import cv2
import numpy as np
from pathlib import Path


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization applied per channel in LAB space.
    Boosts local contrast without over-amplifying noise.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_enhanced = clahe.apply(l_channel)

    enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def gray_world_white_balance(image: np.ndarray) -> np.ndarray:
    """
    Gray World white balance assumption: average color of the scene is gray.
    Corrects the blue-green color cast common in underwater imagery.
    """
    image_float = image.astype(np.float32)

    mean_b = np.mean(image_float[:, :, 0])
    mean_g = np.mean(image_float[:, :, 1])
    mean_r = np.mean(image_float[:, :, 2])
    mean_gray = (mean_b + mean_g + mean_r) / 3.0

    image_float[:, :, 0] = np.clip(image_float[:, :, 0] * (mean_gray / (mean_b + 1e-6)), 0, 255)
    image_float[:, :, 1] = np.clip(image_float[:, :, 1] * (mean_gray / (mean_g + 1e-6)), 0, 255)
    image_float[:, :, 2] = np.clip(image_float[:, :, 2] * (mean_gray / (mean_r + 1e-6)), 0, 255)

    return image_float.astype(np.uint8)


def stretch_histogram(image: np.ndarray, percentile_low: float = 2.0, percentile_high: float = 98.0) -> np.ndarray:
    """
    Per-channel histogram stretching. Expands dynamic range to [0, 255]
    using the given percentiles to clip outliers before stretching.
    """
    output = np.zeros_like(image)
    for ch in range(3):
        channel = image[:, :, ch].astype(np.float32)
        p_low = np.percentile(channel, percentile_low)
        p_high = np.percentile(channel, percentile_high)
        stretched = (channel - p_low) / (p_high - p_low + 1e-6) * 255.0
        output[:, :, ch] = np.clip(stretched, 0, 255).astype(np.uint8)
    return output


def enhance_underwater(image: np.ndarray, use_white_balance: bool = True, use_clahe: bool = True) -> np.ndarray:
    """
    Full underwater enhancement pipeline: white balance → histogram stretch → CLAHE.
    Apply this before running detection inference.

    Args:
        image: BGR image as numpy array (OpenCV format)
        use_white_balance: apply gray world white balance
        use_clahe: apply CLAHE contrast enhancement

    Returns:
        Enhanced BGR image
    """
    enhanced = image.copy()

    if use_white_balance:
        enhanced = gray_world_white_balance(enhanced)

    enhanced = stretch_histogram(enhanced)

    if use_clahe:
        enhanced = apply_clahe(enhanced)

    return enhanced


def enhance_directory(input_dir: str, output_dir: str, extensions: tuple = ('.jpg', '.jpeg', '.png')) -> None:
    """
    Batch-enhance all images in a directory and save to output_dir.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = [f for f in input_path.rglob('*') if f.suffix.lower() in extensions]
    print(f"Found {len(image_files)} images in {input_dir}")

    for img_file in image_files:
        image = cv2.imread(str(img_file))
        if image is None:
            continue
        enhanced = enhance_underwater(image)
        relative = img_file.relative_to(input_path)
        out_file = output_path / relative
        out_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_file), enhanced)

    print(f"Enhanced images saved to {output_dir}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 3:
        enhance_directory(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python preprocess.py <input_dir> <output_dir>")
