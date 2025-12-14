#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import subprocess
import numpy as np
import cv2


class CameraBridgeNode(Node):
    def __init__(self):
        super().__init__("camera_bridge_node")

        self.width = 640
        self.height = 480
        self.channels = 3  # RGB
        self.frame_size = self.width * self.height * self.channels

        self.publisher = self.create_publisher(Image, "/camera/image_raw", 10)

        self.get_logger().info("Starting rpicam-vid...")

        self.proc = subprocess.Popen(
            [
                "rpicam-vid",
                "--codec",
                "rgb",
                "--width",
                str(self.width),
                "--height",
                str(self.height),
                "--timeout",
                "0",
                "-o",
                "-",
            ],
            stdout=subprocess.PIPE,
            bufsize=self.frame_size,
        )

        self.timer = self.create_timer(0.03, self.read_frame)  # ~30 FPS

    def read_frame(self):
        raw = self.proc.stdout.read(self.frame_size)
        if len(raw) != self.frame_size:
            self.get_logger().warn("Incomplete frame received")
            return

        frame = np.frombuffer(raw, dtype=np.uint8)
        frame = frame.reshape((self.height, self.width, 3))

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        msg.height = self.height
        msg.width = self.width
        msg.encoding = "rgb8"
        msg.step = self.width * 3
        msg.data = frame.tobytes()

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CameraBridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
