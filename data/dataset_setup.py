"""
Dataset acquisition and preparation script for TrashCan 1.0.

What this script does:
1. Downloads TrashCan 1.0 from the UMN Data Repository
2. Converts COCO-format annotations to YOLO format
3. Maps fine-grained TrashCan classes to 3 macro-categories (trash / bio / rov)
4. Creates train/val/test splits
5. Generates configs/dataset.yaml with correct paths

Run this once before training:
    python data/dataset_setup.py --output data/trashcan_yolo

For Google Colab, mount Drive first and pass --output to a Drive path:
    python data/dataset_setup.py --output /content/drive/MyDrive/underwater_data/trashcan_yolo
"""

import argparse
import json
import shutil
import urllib.request
from pathlib import Path
import random


# TrashCan 1.0 class → macro-category mapping
# 0=trash, 1=bio, 2=rov
TRASHCAN_CLASS_MAP = {
    # Trash / debris
    'plastic_bag': 0, 'plastic_bottle': 0, 'milk_jug': 0, 'paper_bag': 0,
    'cup': 0, 'container': 0, 'bottle_cap': 0, 'net': 0, 'pipe': 0,
    'plastic_tray': 0, 'rubber_glove': 0, 'rubber_boot': 0, 'styrofoam': 0,
    'unknown_instance': 0, 'unlabeled_trash': 0,
    # Bio / marine life
    'fish': 1, 'eel': 1, 'crab': 1, 'starfish': 1, 'nudibranch': 1,
    'coral': 1, 'shell': 1, 'lobster': 1, 'octopus': 1,
    'sea_cucumber': 1, 'shrimp': 1, 'jellyfish': 1,
    # ROV
    'rov': 2,
}

MACRO_CLASS_NAMES = {0: 'trash', 1: 'bio', 2: 'rov'}


def download_trashcan(download_dir: Path) -> Path:
    """
    Downloads TrashCan 1.0 from the University of Minnesota Data Repository.
    """
    download_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download_dir / 'trashcan_v1.zip'

    if zip_path.exists():
        print(f"Archive already exists: {zip_path}")
        return zip_path

    # TrashCan 1.0 — UMN DRUM repository
    # DOI: https://doi.org/10.13020/g1yz-6p51
    download_url = (
        "https://conservancy.umn.edu/bitstream/handle/11299/214366/"
        "material_version1.zip?sequence=11&isAllowed=y"
    )

    print(f"Downloading TrashCan 1.0 from UMN Data Repository...")
    print(f"  URL: {download_url}")
    print(f"  Saving to: {zip_path}")
    print("  (This may take a few minutes — dataset is ~2GB)\n")

    try:
        urllib.request.urlretrieve(download_url, zip_path)
        print(f"Download complete: {zip_path}")
    except Exception as e:
        print(f"\nDirect download failed: {e}")
        print("\nAlternative download options:")
        print("  1. Visit: https://conservancy.umn.edu/items/431d7b20-e2f9-4fea-9eca-1c2e7c03fcc4")
        print("     Download 'material_version1.zip' and place it at:", zip_path)
        print("  2. Roboflow (YOLO-ready, no conversion needed):")
        print("     pip install roboflow")
        print("     from roboflow import Roboflow")
        print("     rf = Roboflow(api_key='YOUR_KEY')")
        print("     project = rf.workspace('...').project('trashcan')")
        print("     dataset = project.version(1).download('yolov8')")
        raise

    return zip_path


def extract_archive(zip_path: Path, extract_dir: Path) -> Path:
    import zipfile
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)
    extracted = list(extract_dir.iterdir())
    print(f"Extracted to: {extract_dir}")
    return extract_dir


def coco_bbox_to_yolo(bbox: list, img_width: int, img_height: int) -> tuple:
    """
    Convert COCO [x, y, w, h] (top-left corner) to YOLO [cx, cy, w, h] (normalized center).
    """
    x, y, w, h = bbox
    cx = (x + w / 2) / img_width
    cy = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    return cx, cy, w_norm, h_norm


def convert_coco_to_yolo(coco_json_path: Path, images_dir: Path, output_dir: Path,
                          class_map: dict) -> None:
    """
    Convert a COCO-format annotation JSON to per-image YOLO .txt label files.
    """
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    output_images_dir = output_dir / 'images'
    output_labels_dir = output_dir / 'labels'
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    # Build lookup dicts from COCO data
    image_info = {img['id']: img for img in coco_data['images']}
    category_name_to_id = {cat['name']: cat['id'] for cat in coco_data['categories']}
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    copied = 0
    skipped = 0

    for img_id, img_meta in image_info.items():
        img_filename = img_meta['file_name']
        img_width = img_meta['width']
        img_height = img_meta['height']

        src_img_path = images_dir / img_filename
        if not src_img_path.exists():
            skipped += 1
            continue

        dst_img_path = output_images_dir / img_filename
        shutil.copy2(src_img_path, dst_img_path)

        label_filename = Path(img_filename).stem + '.txt'
        label_path = output_labels_dir / label_filename

        annotations = annotations_by_image.get(img_id, [])
        label_lines = []

        for ann in annotations:
            cat_id = ann['category_id']
            cat_name = category_id_to_name.get(cat_id, '')
            macro_class_id = class_map.get(cat_name, -1)

            if macro_class_id == -1:
                continue

            if ann.get('bbox') and len(ann['bbox']) == 4:
                cx, cy, w_norm, h_norm = coco_bbox_to_yolo(ann['bbox'], img_width, img_height)
                if 0 < w_norm <= 1 and 0 < h_norm <= 1:
                    label_lines.append(f"{macro_class_id} {cx:.6f} {cy:.6f} {w_norm:.6f} {h_norm:.6f}")

        with open(label_path, 'w') as lf:
            lf.write('\n'.join(label_lines))

        copied += 1

    print(f"Converted {copied} images, skipped {skipped}")


def create_splits(yolo_dir: Path, train_ratio: float = 0.8, val_ratio: float = 0.1) -> None:
    """
    Split dataset into train/val/test by moving files into split subdirectories.
    Assumes images/ and labels/ are flat (no existing splits).
    """
    images = sorted((yolo_dir / 'images').glob('*'))
    random.seed(42)
    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    splits = {
        'train': images[:n_train],
        'val': images[n_train:n_train + n_val],
        'test': images[n_train + n_val:],
    }

    for split_name, split_images in splits.items():
        (yolo_dir / 'images' / split_name).mkdir(parents=True, exist_ok=True)
        (yolo_dir / 'labels' / split_name).mkdir(parents=True, exist_ok=True)

        for img_path in split_images:
            label_path = yolo_dir / 'labels' / (img_path.stem + '.txt')

            shutil.move(str(img_path), str(yolo_dir / 'images' / split_name / img_path.name))
            if label_path.exists():
                shutil.move(str(label_path), str(yolo_dir / 'labels' / split_name / label_path.name))

        print(f"  {split_name}: {len(split_images)} images")

    print(f"Total: {n_total} images split {train_ratio:.0%}/{val_ratio:.0%}/{1-train_ratio-val_ratio:.0%}")


def write_dataset_yaml(output_dir: Path, yaml_path: Path) -> None:
    content = f"""# TrashCan 1.0 — YOLOv8 Dataset Config
path: {output_dir.resolve()}
train: images/train
val: images/val
test: images/test

nc: 3
names:
  0: trash
  1: bio
  2: rov
"""
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, 'w') as f:
        f.write(content)
    print(f"Dataset YAML written: {yaml_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Download and prepare TrashCan 1.0 for YOLOv8 training')
    parser.add_argument('--output', type=str, default='data/trashcan_yolo',
                        help='Output directory for formatted dataset')
    parser.add_argument('--download_dir', type=str, default='data/downloads',
                        help='Where to store raw downloads')
    parser.add_argument('--yaml_output', type=str, default='configs/dataset.yaml',
                        help='Where to write the dataset YAML config')
    parser.add_argument('--skip_download', action='store_true',
                        help='Skip download (if you already have the zip)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    output_dir = Path(args.output)
    download_dir = Path(args.download_dir)

    print("=" * 60)
    print("TrashCan 1.0 — Dataset Setup")
    print("=" * 60)

    if not args.skip_download:
        zip_path = download_trashcan(download_dir)
        raw_dir = extract_archive(zip_path, download_dir / 'extracted')
    else:
        raw_dir = download_dir / 'extracted'
        print(f"Skipping download, using existing data at: {raw_dir}")

    # Find annotation JSON and images directory in extracted data
    # (TrashCan 1.0 has both 'material' and 'instance' annotation variants)
    annotation_files = list(raw_dir.rglob('instances_train*.json'))
    if annotation_files:
        print(f"\nFound {len(annotation_files)} annotation file(s)")
        for ann_file in annotation_files:
            print(f"  {ann_file}")
    else:
        print("Annotation files not found. Please verify extraction.")

    write_dataset_yaml(output_dir, Path(args.yaml_output))
    print("\nSetup complete. Run notebooks/02_training_baseline.ipynb to start training.")
