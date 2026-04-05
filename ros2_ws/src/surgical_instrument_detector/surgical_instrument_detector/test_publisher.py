"""
test_publisher.py

Simulates a surgical camera by publishing EndoVis frames to /camera/image_raw.
Run this alongside detector_node.py to test the full pipeline.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pathlib import Path
import time

FRAMES_DIR = Path("/mnt/c/Users/Alvin/surgical_cv_project/data/MATIS/endovis_2017/images")
SEQ_PREFIX = "seq2_"
PUBLISH_HZ = 10   # frames per second


class TestImagePublisher(Node):

    def __init__(self):
        super().__init__("test_image_publisher")
        self.pub    = self.create_publisher(Image, "/camera/image_raw", 10)
        self.bridge = CvBridge()

        self.frames = sorted([
            p for p in FRAMES_DIR.glob("*.png")
            if p.name.startswith(SEQ_PREFIX)
        ])
        self.idx = 0
        self.timer = self.create_timer(1.0 / PUBLISH_HZ, self.publish_frame)
        self.get_logger().info(f"Publishing {len(self.frames)} frames at {PUBLISH_HZ}Hz")

    def publish_frame(self):
        if self.idx >= len(self.frames):
            self.idx = 0   # loop

        frame = cv2.imread(str(self.frames[self.idx]))
        msg   = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(msg)
        self.idx += 1


def main(args=None):
    rclpy.init(args=args)
    node = TestImagePublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
