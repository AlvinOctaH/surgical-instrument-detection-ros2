"""
scripts/verify_data.py

Purpose: Verify the EndoVis 2017 dataset before training.
Checks:
  1. Contents of train.json — number of classes, number of files
  2. Whether image & mask files actually exist on disk
  3. Unique pixel values in sampled masks (= classes actually present)
  4. Image size statistics
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
from collections import Counter
import random

# ─── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(r"C:\Users\Alvin\surgical_cv_project\data\MATIS\endovis_2017")
FOLD_DIR  = BASE_DIR / "annotations" / "Fold0"
IMG_DIR   = BASE_DIR / "images"
MASK_DIR  = BASE_DIR / "annotations" / "images"
SAMPLE_N  = 50   # number of masks to sample for pixel value scanning
# ───────────────────────────────────────────────────────────────────────────────


def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def check_coco_json(json_path: Path):
    """
    Read a COCO-format JSON file.
    COCO JSON structure: {images, annotations, categories}
    We extract: class names, number of images, number of annotations.
    """
    print(f"\n{'='*60}")
    print(f"Checking: {json_path.name}")
    print(f"{'='*60}")

    data = load_json(json_path)

    # 1. Classes
    categories = data.get("categories", [])
    print(f"\n[CLASSES] Number of classes: {len(categories)}")
    for cat in categories:
        print(f"  ID {cat['id']:>3} -> {cat['name']}")

    # 2. Images
    images = data.get("images", [])
    print(f"\n[IMAGES] Number of entries: {len(images)}")
    if images:
        print(f"  Example entry: {images[0]}")

    # 3. Annotations
    annotations = data.get("annotations", [])
    print(f"\n[ANNOTATIONS] Number of entries: {len(annotations)}")

    return data


def check_files_exist(data: dict, split_name: str):
    """
    For every image listed in the JSON,
    check whether the file actually exists on disk.
    """
    print(f"\n{'='*60}")
    print(f"Checking file existence for: {split_name}")
    print(f"{'='*60}")

    images = data.get("images", [])
    missing_img  = []
    missing_mask = []

    for entry in images:
        # 'file_name' in COCO can be a relative path or just a filename
        fname = Path(entry["file_name"]).name   # strip any leading path

        img_path  = IMG_DIR  / fname
        mask_path = MASK_DIR / fname

        if not img_path.exists():
            missing_img.append(str(img_path))
        if not mask_path.exists():
            missing_mask.append(str(mask_path))

    print(f"  Total entries  : {len(images)}")
    print(f"  Missing images : {len(missing_img)}")
    print(f"  Missing masks  : {len(missing_mask)}")

    for p in missing_img[:5]:
        print(f"    x IMG  {p}")
    for p in missing_mask[:5]:
        print(f"    x MASK {p}")

    return len(missing_img) == 0 and len(missing_mask) == 0


def scan_mask_pixel_values(data: dict):
    """
    Open a random sample of masks and collect all unique pixel values.
    This confirms which classes are ACTUALLY present in the data
    vs. what is merely defined in the JSON.

    Masks are stored as grayscale PNGs — pixel value = class ID.
    """
    print(f"\n{'='*60}")
    print("Scanning mask pixel values (random sample)")
    print(f"{'='*60}")

    images = data.get("images", [])
    sample = random.sample(images, min(SAMPLE_N, len(images)))

    all_values = Counter()
    failed     = 0
    sizes      = []

    for entry in sample:
        fname     = Path(entry["file_name"]).name
        mask_path = MASK_DIR / fname

        try:
            mask = np.array(Image.open(mask_path))
            sizes.append(mask.shape)

            # np.unique() returns every distinct pixel value in this mask
            for v in np.unique(mask):
                all_values[v] += 1   # count how many masks contain this value

        except Exception as e:
            print(f"  Failed to read {mask_path.name}: {e}")
            failed += 1

    print(f"\n  Scanned : {len(sample) - failed} masks")
    print(f"  Failed  : {failed}")

    print(f"\n  Unique pixel values found (class IDs present in data):")
    for val, count in sorted(all_values.items()):
        print(f"    Value {val:>3} -> found in {count:>3}/{len(sample)} masks")

    if sizes:
        print(f"\n  Mask sizes found: {set(sizes)}")

    return sorted(all_values.keys())


def main():
    print("EndoVis 2017 — Dataset Verification Script")
    print(f"Base dir: {BASE_DIR}\n")

    # Check whether all key directories exist
    for d in [BASE_DIR, IMG_DIR, MASK_DIR, FOLD_DIR]:
        status = "OK" if d.exists() else "MISSING"
        print(f"  [{status}]  {d}")

    train_json_path = FOLD_DIR / "train.json"
    val_json_path   = FOLD_DIR / "val.json"

    if not train_json_path.exists():
        print(f"\nERROR: {train_json_path} not found!")
        return

    # Check JSON contents
    train_data = check_coco_json(train_json_path)
    val_data   = check_coco_json(val_json_path) if val_json_path.exists() else None

    # Check file existence on disk
    train_ok = check_files_exist(train_data, "train")
    if val_data:
        val_ok = check_files_exist(val_data, "val")

    # Scan actual pixel values in masks
    found_classes = scan_mask_pixel_values(train_data)

    # Final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Classes defined in JSON : {len(train_data.get('categories', []))}")
    print(f"  Class IDs found in masks: {found_classes}")
    print(f"  File integrity          : {'OK' if train_ok else 'MISSING FILES DETECTED'}")
    print(f"\nReady to proceed to dataset.py if everything looks good.")


if __name__ == "__main__":
    main()