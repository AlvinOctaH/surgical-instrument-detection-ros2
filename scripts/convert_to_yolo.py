"""
scripts/convert_to_yolo.py

Converts EndoVis 2017 semantic segmentation masks → YOLO instance segmentation format.

YOLO label format (one line per instance):
    class_id x1 y1 x2 y2 ... xn yn
    - class_id is 0-indexed (so EndoVis class 1 → YOLO class 0)
    - coordinates are normalized [0.0, 1.0] relative to image size
    - polygon points are the contour of the instance mask

Output structure:
    data/yolo_dataset/
        images/
            train/   ← symlinks or copies of PNG frames
            val/
        labels/
            train/   ← .txt files, one per frame
            val/
        dataset.yaml ← YOLO config file
"""

import json
import numpy as np
import cv2
from pathlib import Path
import shutil
from tqdm import tqdm
from PIL import Image

# ─── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(r"C:\Users\Alvin\surgical_cv_project\data\MATIS\endovis_2017")
OUT_DIR     = Path(r"C:\Users\Alvin\surgical_cv_project\data\yolo_dataset")
FOLD_DIR    = BASE_DIR / "annotations" / "Fold0"
IMG_DIR     = BASE_DIR / "images"
MASK_DIR    = BASE_DIR / "annotations" / "images"

# Minimum contour area in pixels — ignore tiny noise regions
MIN_AREA    = 100
# ───────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "Bipolar Forceps",       # YOLO class 0  (was EndoVis class 1)
    "Prograsp Forceps",      # YOLO class 1
    "Large Needle Driver",   # YOLO class 2
    "Vessel Sealer",         # YOLO class 3
    "Grasping Retractor",    # YOLO class 4
    "Monopolar Curved Scissors",  # YOLO class 5
    "Ultrasound Probe",      # YOLO class 6
]


def mask_to_yolo_labels(mask: np.ndarray, img_h: int, img_w: int) -> list[str]:
    """
    Convert one semantic mask → list of YOLO label strings.

    Steps for each class:
      1. Binary mask: pixels where mask == class_id
      2. findContours: get polygon outlines of each connected region
      3. Filter small regions (noise)
      4. Normalize coordinates and format as YOLO string

    Args:
        mask  : 2D numpy array, dtype uint8, values 0–7
        img_h : image height in pixels
        img_w : image width in pixels

    Returns:
        List of strings, one per instrument instance detected
    """
    lines = []

    for endovis_class_id in range(1, 8):        # 1–7
        yolo_class_id = endovis_class_id - 1    # 0–6

        # Step 1: isolate pixels of this class
        binary = (mask == endovis_class_id).astype(np.uint8) * 255

        if binary.max() == 0:
            continue  # this class not present in frame, skip

        # Step 2: find contours
        # RETR_EXTERNAL: only outer contours (ignore holes)
        # CHAIN_APPROX_SIMPLE: compress straight segments (fewer points)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            # Step 3: filter tiny regions
            area = cv2.contourArea(contour)
            if area < MIN_AREA:
                continue

            # contour shape: (N, 1, 2) → flatten to (N, 2)
            points = contour.reshape(-1, 2)

            # Need at least 3 points to form a valid polygon
            if len(points) < 3:
                continue

            # Step 4: normalize to [0, 1]
            points_normalized = points.astype(float)
            points_normalized[:, 0] /= img_w   # x
            points_normalized[:, 1] /= img_h   # y

            # Clamp to [0, 1] — safety against edge pixels
            points_normalized = np.clip(points_normalized, 0.0, 1.0)

            # Format: "class_id x1 y1 x2 y2 ..."
            coords_str = " ".join(
                f"{x:.6f} {y:.6f}" for x, y in points_normalized
            )
            lines.append(f"{yolo_class_id} {coords_str}")

    return lines


def process_split(json_path: Path, split_name: str):
    """Process one split (train or val)."""
    print(f"\n--- Processing {split_name} split ---")

    with open(json_path) as f:
        data = json.load(f)

    # Output directories
    out_img_dir   = OUT_DIR / "images"  / split_name
    out_label_dir = OUT_DIR / "labels" / split_name
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    img_h, img_w = 1024, 1280   # verified from dataset scan

    stats = {"total": 0, "with_labels": 0, "empty": 0, "instances": 0}

    for entry in tqdm(data["images"], desc=split_name):
        fname    = Path(entry["file_name"]).name
        img_src  = IMG_DIR  / fname
        mask_src = MASK_DIR / fname

        if not img_src.exists() or not mask_src.exists():
            print(f"  Skipping missing: {fname}")
            continue

        # Read mask as grayscale — pixel values ARE the class IDs
        mask = np.array(Image.open(mask_src))

        # Convert mask → YOLO label lines
        label_lines = mask_to_yolo_labels(mask, img_h, img_w)

        # Write .txt label file (same stem as image)
        label_path = out_label_dir / (Path(fname).stem + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(label_lines))

        # Copy image to YOLO structure
        shutil.copy2(img_src, out_img_dir / fname)

        # Stats
        stats["total"] += 1
        if label_lines:
            stats["with_labels"] += 1
            stats["instances"]   += len(label_lines)
        else:
            stats["empty"] += 1

    print(f"  Processed  : {stats['total']} frames")
    print(f"  With labels: {stats['with_labels']} frames")
    print(f"  Empty      : {stats['empty']} frames (background only)")
    print(f"  Instances  : {stats['instances']} total")


def write_dataset_yaml():
    """Write YOLO dataset config file."""
    yaml_content = f"""# EndoVis 2017 — YOLOv8 instance segmentation dataset
path: {OUT_DIR.as_posix()}
train: images/train
val:   images/val

nc: 7
names:
  0: Bipolar Forceps
  1: Prograsp Forceps
  2: Large Needle Driver
  3: Vessel Sealer
  4: Grasping Retractor
  5: Monopolar Curved Scissors
  6: Ultrasound Probe
"""
    yaml_path = OUT_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"\nDataset YAML written: {yaml_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    process_split(FOLD_DIR / "train.json", "train")
    process_split(FOLD_DIR / "val.json",   "val")
    write_dataset_yaml()

    print("\n✓ Conversion complete. Ready for dataset.py and training.")


if __name__ == "__main__":
    main()