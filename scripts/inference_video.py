"""
scripts/inference_video.py

Run YOLO11s-seg + ByteTrack on a sequence of frames.
Outputs a demo video with:
  - Instance segmentation masks (colored per class)
  - Bounding boxes with persistent track IDs
  - Class labels + confidence scores
  - Frame counter overlay

ByteTrack is built into Ultralytics — no separate install needed.
model.track() handles detection + tracking in one call.
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ─── CONFIG ────────────────────────────────────────────────────────────────────
WEIGHTS    = Path(r"C:\Users\Alvin\surgical_cv_project\results\endovis_yolo11s_seg_v1\weights\best.pt")
FRAMES_DIR = Path(r"C:\Users\Alvin\surgical_cv_project\data\MATIS\endovis_2017\images")
OUT_VIDEO  = Path(r"C:\Users\Alvin\surgical_cv_project\results\demo_tracking.mp4")

# Use seq2 — it has Prograsp Forceps + Large Needle Driver, visually rich
SEQ_PREFIX = "seq2_"

CONF_THRESH = 0.25
IOU_THRESH  = 0.45
IMG_SIZE    = 640
FPS         = 15
# ───────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver",
    "Vessel Sealer", "Grasping Retractor", "Monopolar Curved Scissors",
    "Ultrasound Probe",
]

CLASS_COLORS = [
    (74,  222, 128), (251, 146,  60), (96,  165, 250),
    (248, 113, 113), (167, 139, 250), (34,  211, 238),
    (250, 204,  21),
]


def draw_mask_overlay(frame: np.ndarray, mask: np.ndarray, color: tuple, alpha: float = 0.4):
    """Blend a binary mask onto the frame with transparency."""
    colored = np.zeros_like(frame)
    colored[:] = color
    # Where mask is True, blend colored layer over frame
    frame[mask] = (alpha * np.array(color) + (1 - alpha) * frame[mask]).astype(np.uint8)


def draw_label(frame, text, x, y, color):
    """Draw a filled badge with text."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 6, y), color, -1)
    cv2.putText(frame, text, (x + 3, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def main():
    OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    # Collect frames for chosen sequence, sorted by frame number
    frames = sorted([
        p for p in FRAMES_DIR.glob("*.png")
        if p.name.startswith(SEQ_PREFIX)
    ])
    print(f"Found {len(frames)} frames for sequence: {SEQ_PREFIX}")

    if not frames:
        print("No frames found. Check SEQ_PREFIX.")
        return

    # Read one frame to get dimensions
    sample = cv2.imread(str(frames[0]))
    h, w   = sample.shape[:2]

    # VideoWriter — mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT_VIDEO), fourcc, FPS, (w, h))

    model = YOLO(str(WEIGHTS))

    print(f"Running inference + ByteTrack on {len(frames)} frames...")

    for frame_idx, frame_path in enumerate(frames):
        frame = cv2.imread(str(frame_path))

        # model.track() = detection + ByteTrack tracking in one call
        # persist=True tells tracker to maintain state across frames
        results = model.track(
            source   = frame,
            conf     = CONF_THRESH,
            iou      = IOU_THRESH,
            imgsz    = IMG_SIZE,
            tracker  = "bytetrack.yaml",   # built-in ByteTrack config
            persist  = True,               # maintain track IDs across calls
            verbose  = False,
        )

        vis = frame.copy()
        result = results[0]

        if result.masks is not None and result.boxes is not None:
            masks  = result.masks.data.cpu().numpy()    # (N, H', W') binary
            boxes  = result.boxes
            cls_ids   = boxes.cls.cpu().numpy().astype(int)
            confs     = boxes.conf.cpu().numpy()
            track_ids = boxes.id                        # None if no tracks yet

            for i in range(len(cls_ids)):
                cid   = cls_ids[i]
                conf  = confs[i]
                color = CLASS_COLORS[cid % len(CLASS_COLORS)]
                tid   = int(track_ids[i]) if track_ids is not None else -1

                # Resize mask from model output size → original frame size
                mask_resized = cv2.resize(
                    masks[i].astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)

                # Draw semi-transparent mask fill
                draw_mask_overlay(vis, mask_resized, color, alpha=0.35)

                # Draw polygon outline
                contours, _ = cv2.findContours(
                    mask_resized.astype(np.uint8),
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(vis, contours, -1, color, 2)

                # Draw label: "ClassName #TrackID (conf)"
                x1, y1 = int(boxes.xyxy[i][0]), int(boxes.xyxy[i][1])
                label = f"{CLASS_NAMES[cid]} #{tid} {conf:.2f}"
                draw_label(vis, label, x1, max(y1, 20), color)

        # Frame counter (bottom-left)
        cv2.putText(vis, f"Frame {frame_idx+1:03d}/{len(frames)}",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1, cv2.LINE_AA)

        writer.write(vis)

        if (frame_idx + 1) % 50 == 0:
            print(f"  Processed {frame_idx+1}/{len(frames)} frames")

    writer.release()
    print(f"\nDemo video saved: {OUT_VIDEO}")
    print(f"Duration: {len(frames)/FPS:.1f}s at {FPS}fps")


if __name__ == "__main__":
    main()