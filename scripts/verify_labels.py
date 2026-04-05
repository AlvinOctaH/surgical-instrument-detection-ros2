"""
scripts/verify_labels.py

Visual verification of converted YOLO labels.
Saves overlay images so you can inspect them without a display.
"""

import sys
import random
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dataset import YOLODataset

YOLO_DIR  = Path(r"C:\Users\Alvin\surgical_cv_project\data\yolo_dataset")
OUT_DIR   = Path(r"C:\Users\Alvin\surgical_cv_project\results\predictions")
N_SAMPLES = 8   # how many frames to visualize


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ds = YOLODataset(YOLO_DIR, split="train")

    # ── Class distribution ──────────────────────────────────────────────────
    print("\nClass distribution (train split):")
    dist = ds.class_distribution()
    total = sum(dist.values())
    for name, count in dist.items():
        bar = "█" * int(30 * count / max(dist.values()))
        print(f"  {name:<30} {count:>5}  {bar}")
    print(f"  {'TOTAL':<30} {total:>5}")

    # ── Visual check ────────────────────────────────────────────────────────
    print(f"\nSaving {N_SAMPLES} overlay images to {OUT_DIR}")
    indices = random.sample(range(len(ds)), N_SAMPLES)

    for i, idx in enumerate(indices):
        vis = ds.draw_labels(idx)
        fname = ds.samples[idx].name
        out_path = OUT_DIR / f"verify_{i:02d}_{fname}"
        cv2.imwrite(str(out_path), vis)
        print(f"  Saved: {out_path.name}")

    print("\nOpen the saved images to inspect polygon alignment.")
    print("What to look for:")
    print("  ✓ Polygons tightly follow instrument outlines")
    print("  ✓ Labels show correct class names")
    print("  ✗ If polygons are boxes or blobs → contour approx issue")
    print("  ✗ If polygons are offset/scaled → normalization bug")


if __name__ == "__main__":
    main()