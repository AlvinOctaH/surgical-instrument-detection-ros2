# Surgical Instrument Detection & Tracking with ROS2

Real-time surgical instrument instance segmentation and tracking pipeline built on EndoVis 2017, YOLO11s-seg, ByteTrack, and ROS2 Humble. Designed for integration with surgical robotic platforms (dVRK).

![Python](https://img.shields.io/badge/Python-3.10-blue)
![ROS2](https://img.shields.io/badge/ROS2-Humble-orange)
![Framework](https://img.shields.io/badge/Framework-Ultralytics_YOLO11-purple)
![Dataset](https://img.shields.io/badge/Dataset-EndoVis_2017-green)
![GPU](https://img.shields.io/badge/GPU-RTX_500_Ada_4GB-red)

---

## Demo

> Tracking output on EndoVis 2017 seq2 — each instrument gets a persistent track ID across frames.

*(Insert GIF or video screenshot here — export a short clip from `results/demo_tracking.mp4`)*

---

## Results

Evaluated on EndoVis 2017 Fold 0 validation split (450 frames, 4 classes present).

| Class | Box mAP50 | Mask mAP50 | Mask mAP50-95 |
|---|---|---|---|
| **All (mean)** | 0.422 | 0.468 | 0.294 |
| Large Needle Driver | 0.684 | 0.709 | 0.569 |
| Bipolar Forceps | 0.316 | 0.404 | 0.231 |
| Prograsp Forceps | 0.329 | 0.373 | 0.170 |
| Ultrasound Probe | 0.361 | 0.387 | 0.206 |

> **Note:** Vessel Sealer, Grasping Retractor, and Monopolar Curved Scissors are absent from the Fold 0 validation split — evaluation for these classes requires full 4-fold cross-validation.

**Runtime:** ~6 Hz on NVIDIA RTX 500 Ada Laptop GPU (4 GB VRAM) via ROS2 node.

---

## Pipeline Overview

```
EndoVis 2017 Semantic Masks
        │
        ▼
scripts/convert_to_yolo.py      ← semantic PNG → YOLO instance polygon .txt
        │
        ▼
scripts/train.py                ← YOLO11s-seg fine-tuning (100 epochs, AdamW)
        │
        ▼
scripts/inference_video.py      ← YOLO11s-seg + ByteTrack → demo .mp4
        │
        ▼
ROS2 surgical_instrument_detector
  ├── detector_node.py          ← subscribes /camera/image_raw
  ├── publishes /surgical/detections          (vision_msgs/Detection2DArray)
  └── publishes /surgical/annotated_image     (sensor_msgs/Image)
```

---

## Project Structure

```
surgical_cv_project/
├── src/
│   └── dataset.py              # YOLO dataset loader + visualization utilities
├── scripts/
│   ├── convert_to_yolo.py      # Convert EndoVis semantic masks → YOLO format
│   ├── verify_labels.py        # Verify converted labels with polygon overlays
│   ├── train.py                # YOLO11s-seg training script
│   └── inference_video.py      # Inference + ByteTrack demo video
├── ros2_ws/
│   └── src/surgical_instrument_detector/
│       ├── detector_node.py    # ROS2 detector + tracker node
│       └── test_publisher.py   # Simulated camera publisher (for testing)
├── results/
│   ├── predictions/            # Verification overlay images
│   └── demo_tracking.mp4       # Tracking demo video
└── data/                       # Not tracked (see Dataset section)
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/surgical-instrument-detection.git
cd surgical-instrument-detection

conda create -n surgical_cv python=3.10
conda activate surgical_cv
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics opencv-python numpy matplotlib pandas tqdm wandb PyYAML
```

### 2. Dataset

Download the EndoVis 2017 dataset via the [MATIS bundle](https://github.com/BCV-Uniandes/MATIS).
Extract to `data/MATIS/endovis_2017/` with the following structure:

```
endovis_2017/
├── images/               ← 1800 PNG frames
└── annotations/
    ├── images/           ← 1800 semantic mask PNGs
    └── Fold0/
        ├── train.json
        └── val.json
```

### 3. Convert annotations

```bash
python scripts/convert_to_yolo.py
```

Converts semantic masks → YOLO instance segmentation format.
Output: `data/yolo_dataset/` with YOLO-compatible images, labels, and `dataset.yaml`.

### 4. Verify labels

```bash
python scripts/verify_labels.py
```

Saves polygon overlay images to `results/predictions/` for visual inspection.

### 5. Train

```bash
python scripts/train.py
```

Trains YOLO11s-seg for 100 epochs. Weights saved to `results/endovis_yolo11s_seg_v1/weights/best.pt`.

### 6. Demo video

```bash
python scripts/inference_video.py
```

Runs YOLO11s-seg + ByteTrack on seq2 frames. Output: `results/demo_tracking.mp4`.

---

## ROS2 Integration

Tested on Ubuntu 22.04 / ROS2 Humble.

### Install dependencies

```bash
sudo apt install -y ros-humble-vision-msgs ros-humble-cv-bridge ros-humble-rqt-image-view
pip3 install ultralytics
```

### Build

```bash
cd ros2_ws
colcon build --packages-select surgical_instrument_detector
source install/setup.bash
```

### Run

Terminal 1 — detector node:
```bash
source install/setup.bash
ros2 run surgical_instrument_detector detector_node
```

Terminal 2 — simulated camera (uses EndoVis frames):
```bash
source install/setup.bash
ros2 run surgical_instrument_detector test_publisher
```

Terminal 3 — visualize:
```bash
ros2 run rqt_image_view rqt_image_view
# Select /surgical/annotated_image in the dropdown
```

### Published Topics

| Topic | Type | Description |
|---|---|---|
| `/surgical/detections` | `vision_msgs/Detection2DArray` | Per-instance class, confidence, bbox, track ID |
| `/surgical/annotated_image` | `sensor_msgs/Image` | Visualization frame with overlays |

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `weights_path` | `...best.pt` | Path to model weights |
| `conf_threshold` | `0.25` | Detection confidence threshold |
| `iou_threshold` | `0.45` | NMS IoU threshold |
| `img_size` | `640` | Inference image size |

Override at launch:
```bash
ros2 run surgical_instrument_detector detector_node \
  --ros-args -p conf_threshold:=0.3 -p img_size:=480
```

---

## Training Details

| Setting | Value |
|---|---|
| Model | YOLO11s-seg (pretrained COCO) |
| Dataset | EndoVis 2017, Fold 0 |
| Train / Val | 1350 / 450 frames |
| Epochs | 100 |
| Optimizer | AdamW (lr=0.001) |
| Batch size | 8 |
| Image size | 640 × 640 |
| GPU | NVIDIA RTX 500 Ada (4 GB) |
| Training time | ~6 hours |

---

## Limitations

- Single-fold evaluation (Fold 0 only) — 3 of 7 classes have no validation instances in this split. Full 4-fold cross-validation is needed for complete class coverage.
- ~6 Hz throughput on mobile GPU — below real-time requirements for closed-loop control. TensorRT quantization expected to resolve this.
- Connected same-class instances are merged into one polygon — learned instance separation would handle instrument contact better.

---

## References

- EndoVis 2017: MICCAI 2017 Robotic Instrument Segmentation Challenge
- YOLO11: Ultralytics, 2024
- ByteTrack: Zhang et al., ECCV 2022
- MATIS: BCV-Uniandes, 2022

---

## Author

**Alvin** — B.Eng. Robotics & AI
Intern, Dept. Biomedical Engineering, NCKU Taiwan
