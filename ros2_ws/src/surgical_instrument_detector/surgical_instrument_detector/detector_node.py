"""
detector_node.py

ROS2 node: surgical instrument detection + tracking.

Subscribes:
    /camera/image_raw  (sensor_msgs/Image)

Publishes:
    /surgical/detections      (vision_msgs/Detection2DArray)
    /surgical/annotated_image (sensor_msgs/Image)

To test without a real camera, run the companion image publisher:
    python3 test_publisher.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from std_msgs.msg import Header

import cv2
import numpy as np
from pathlib import Path

# cv_bridge converts between ROS Image messages and OpenCV arrays
from cv_bridge import CvBridge

# Ultralytics must be importable inside WSL2
# If not: pip3 install ultralytics
from ultralytics import YOLO


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


class SurgicalDetectorNode(Node):

    def __init__(self):
        super().__init__("surgical_detector")

        # ── Parameters (can be overridden at launch) ──────────────────────
        self.declare_parameter("weights_path", "")
        self.declare_parameter("conf_threshold", 0.25)
        self.declare_parameter("iou_threshold",  0.45)
        self.declare_parameter("img_size",       640)

        weights = self.get_parameter("weights_path").get_parameter_value().string_value
        self.conf   = self.get_parameter("conf_threshold").get_parameter_value().double_value
        self.iou    = self.get_parameter("iou_threshold").get_parameter_value().double_value
        self.imgsz  = self.get_parameter("img_size").get_parameter_value().integer_value

        if not weights:
            # Default path — adjust to where your weights are inside WSL2
            weights = "/mnt/c/Users/Alvin/surgical_cv_project/results/endovis_yolo11s_seg_v1/weights/best.pt"

        self.get_logger().info(f"Loading model from: {weights}")
        self.model  = YOLO(weights)
        self.bridge = CvBridge()

        # ── Subscriber ────────────────────────────────────────────────────
        self.sub = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.image_callback,
            10   # QoS queue depth
        )

        # ── Publishers ────────────────────────────────────────────────────
        self.det_pub = self.create_publisher(
            Detection2DArray,
            "/surgical/detections",
            10
        )
        self.img_pub = self.create_publisher(
            Image,
            "/surgical/annotated_image",
            10
        )

        self.frame_count = 0
        self.get_logger().info("SurgicalDetectorNode ready. Waiting for images...")

    def image_callback(self, msg: Image):
        """Called every time a new frame arrives on /camera/image_raw."""

        # Convert ROS Image message → OpenCV BGR array
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        h, w  = frame.shape[:2]

        # Run YOLO + ByteTrack
        results = self.model.track(
            source  = frame,
            conf    = self.conf,
            iou     = self.iou,
            imgsz   = self.imgsz,
            tracker = "bytetrack.yaml",
            persist = True,
            verbose = False,
        )

        result    = results[0]
        det_array = Detection2DArray()
        det_array.header = msg.header   # same timestamp as input image
        vis = frame.copy()

        if result.boxes is not None and len(result.boxes):
            cls_ids   = result.boxes.cls.cpu().numpy().astype(int)
            confs     = result.boxes.conf.cpu().numpy()
            xyxy      = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id

            masks = result.masks.data.cpu().numpy() if result.masks is not None else None

            for i in range(len(cls_ids)):
                cid   = cls_ids[i]
                conf  = float(confs[i])
                color = CLASS_COLORS[cid % len(CLASS_COLORS)]
                tid   = int(track_ids[i]) if track_ids is not None else -1
                x1, y1, x2, y2 = xyxy[i]

                # ── Build Detection2D message ──────────────────────────
                det = Detection2D()
                det.header = msg.header

                # Bounding box center + size (ROS convention)
                det.bbox.center.position.x = float((x1 + x2) / 2)
                det.bbox.center.position.y = float((y1 + y2) / 2)
                det.bbox.size_x = float(x2 - x1)
                det.bbox.size_y = float(y2 - y1)

                # Class hypothesis
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = str(cid)
                hyp.hypothesis.score    = conf
                det.results.append(hyp)

                # Store track ID in id field
                det.id = str(tid)

                det_array.detections.append(det)

                # ── Visualization ──────────────────────────────────────
                if masks is not None:
                    mask_resized = cv2.resize(
                        masks[i].astype(np.uint8), (w, h),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                    overlay = vis.copy()
                    overlay[mask_resized] = (
                        0.4 * np.array(color) + 0.6 * overlay[mask_resized]
                    ).astype(np.uint8)
                    vis = overlay

                    contours, _ = cv2.findContours(
                        mask_resized.astype(np.uint8),
                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(vis, contours, -1, color, 2)

                label = f"{CLASS_NAMES[cid]} #{tid} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(vis, (int(x1), int(y1) - th - 8),
                              (int(x1) + tw + 6, int(y1)), color, -1)
                cv2.putText(vis, label, (int(x1) + 3, int(y1) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Publish detections
        self.det_pub.publish(det_array)

        # Publish annotated image
        ann_msg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
        ann_msg.header = msg.header
        self.img_pub.publish(ann_msg)

        self.frame_count += 1
        if self.frame_count % 30 == 0:
            n = len(det_array.detections)
            self.get_logger().info(f"Frame {self.frame_count} | {n} detections")


def main(args=None):
    rclpy.init(args=args)
    node = SurgicalDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
