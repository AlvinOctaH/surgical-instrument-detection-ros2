"""
scripts/train.py

YOLOv11-seg training on EndoVis 2017.

Run from project root:
    python scripts/train.py
"""

from ultralytics import YOLO
from pathlib import Path

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_YAML = Path(r"C:\Users\Alvin\surgical_cv_project\data\yolo_dataset\dataset.yaml")
RUN_NAME = "endovis_yolo11s_seg_v1"
PROJECT_DIR  = Path(r"C:\Users\Alvin\surgical_cv_project\results")
# ───────────────────────────────────────────────────────────────────────────────

def main():
    # Load pretrained YOLOv8s-seg weights (downloads ~22MB on first run)
    # Pretrained on COCO — gives us a strong starting point, especially
    # for polygon head weights even though surgical domain differs
    model = YOLO("yolo11s-seg.pt")

    results = model.train(
    data      = str(DATASET_YAML),
    epochs    = 100,           # was 50
    imgsz     = 640,
    batch     = 8,
    device    = 0,
    project   = str(PROJECT_DIR),
    name      = "endovis_yolo11s_seg_v2",   # new run name

    hsv_h     = 0.015,
    hsv_s     = 0.4,
    hsv_v     = 0.3,
    fliplr    = 0.5,
    flipud    = 0.0,
    degrees   = 10.0,
    scale     = 0.3,
    mosaic    = 0.5,

    optimizer     = "AdamW",
    lr0           = 0.001,
    lrf           = 0.01,
    warmup_epochs = 3,
    patience      = 0,         # disable early stopping — let it run full 100

    plots       = True,
    save        = True,
    save_period = 10,
    val         = True,

    overlap_mask = True,
    mask_ratio   = 4,
    )

    print(f"\nTraining complete.")
    print(f"Best weights: {PROJECT_DIR / RUN_NAME / 'weights' / 'best.pt'}")
    print(f"Results:      {PROJECT_DIR / RUN_NAME}")


if __name__ == "__main__":
    main()