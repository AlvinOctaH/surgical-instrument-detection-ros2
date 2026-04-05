"""
src/dataset.py

Utility class for loading and visualizing the converted YOLO dataset.
NOT used by YOLOv8 training — Ultralytics handles that internally.
Used for: visualization, sanity checks, class distribution stats.
"""

import numpy as np
import cv2
from pathlib import Path


CLASS_NAMES = [
    "Bipolar Forceps",
    "Prograsp Forceps",
    "Large Needle Driver",
    "Vessel Sealer",
    "Grasping Retractor",
    "Monopolar Curved Scissors",
    "Ultrasound Probe",
]

# One distinct BGR color per class — used for polygon overlays
CLASS_COLORS = [
    (74,  222, 128),   # green
    (96,  165, 250),   # blue
    (251, 191,  36),   # amber
    (248, 113, 113),   # red
    (167, 139, 250),   # purple
    (34,  211, 238),   # cyan
    (251, 146,  60),   # orange
]


class YOLODataset:
    """
    Loads image + label pairs from a YOLO-format directory.

    Directory structure expected:
        root/
            images/train/   ← .png files
            labels/train/   ← .txt files, same stem as images
    """

    def __init__(self, root: str | Path, split: str = "train"):
        self.root      = Path(root)
        self.split     = split
        self.img_dir   = self.root / "images" / split
        self.label_dir = self.root / "labels" / split

        # Collect all image paths that also have a label file
        self.samples = sorted([
            p for p in self.img_dir.glob("*.png")
            if (self.label_dir / (p.stem + ".txt")).exists()
        ])

        print(f"[YOLODataset] Split={split} | Found {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def load_sample(self, idx: int) -> tuple[np.ndarray, list[dict]]:
        """
        Load one image and its labels.

        Returns:
            img    : HxWx3 BGR numpy array (OpenCV format)
            labels : list of dicts, each with keys:
                       'class_id'  : int (0-indexed)
                       'class_name': str
                       'polygon'   : (N, 2) float array, normalized [0,1]
        """
        img_path   = self.samples[idx]
        label_path = self.label_dir / (img_path.stem + ".txt")

        img = cv2.imread(str(img_path))

        labels = []
        with open(label_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts     = line.split()
                class_id  = int(parts[0])
                # remaining values are x1 y1 x2 y2 ... interleaved
                coords    = list(map(float, parts[1:]))
                # reshape to (N, 2) — each row is (x, y)
                polygon   = np.array(coords).reshape(-1, 2)

                labels.append({
                    "class_id"  : class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "polygon"   : polygon,   # normalized
                })

        return img, labels

    def draw_labels(self, idx: int) -> np.ndarray:
        """
        Draw polygon overlays on the image for visual verification.

        Denormalization:
            pixel_x = norm_x * image_width
            pixel_y = norm_y * image_height
        This is the inverse of what convert_to_yolo.py did.
        """
        img, labels = self.load_sample(idx)
        overlay     = img.copy()
        h, w        = img.shape[:2]

        for inst in labels:
            cid   = inst["class_id"]
            color = CLASS_COLORS[cid]

            # Denormalize: float [0,1] → int pixel coordinates
            poly_px = (inst["polygon"] * np.array([w, h])).astype(np.int32)

            # Semi-transparent fill
            cv2.fillPoly(overlay, [poly_px], color)

            # Solid outline
            cv2.polylines(img, [poly_px], isClosed=True, color=color, thickness=2)

            # Label badge at top-left of polygon bounding box
            x_min = poly_px[:, 0].min()
            y_min = poly_px[:, 1].min()
            label = inst["class_name"]
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (x_min, y_min - th - 6), (x_min + tw + 4, y_min), color, -1)
            cv2.putText(img, label, (x_min + 2, y_min - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # Blend fill (30% opacity) over original
        result = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
        return result

    def class_distribution(self) -> dict[str, int]:
        """
        Count total instances per class across all samples in this split.
        Useful to confirm class imbalance before training.
        """
        counts = {name: 0 for name in CLASS_NAMES}

        for img_path in self.samples:
            label_path = self.label_dir / (img_path.stem + ".txt")
            with open(label_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        cid = int(line.split()[0])
                        counts[CLASS_NAMES[cid]] += 1

        return counts