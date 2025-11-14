#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI Visualization Node (ROS2 Jazzy)
- Subscribe: /camera/image_raw (sensor_msgs/Image)
- Subscribe: /detections (vision_msgs/Detection2DArray)
- Vẽ bounding box + label (class name) + confidence
- Publish:
    /camera/ai_annotated           -> sensor_msgs/Image (cv_bridge)
    /camera/ai_annotated/compressed-> sensor_msgs/CompressedImage (JPEG)
- Ý tưởng: PerceptionNode chạy YOLO và publish detections; node này chỉ vẽ và stream.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

# Messages
from sensor_msgs.msg import Image, CompressedImage
from vision_msgs.msg import Detection2DArray, Detection2D
from std_msgs.msg import Header

# cv bridge và OpenCV
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import threading
import time


class AIVisualizationNode(Node):
    def __init__(self):
        # Khởi tạo node ROS với tên "ai_visualization_node"
        super().__init__("ai_visualization_node")

        # ---------------------------
        # Parameters (có thể thay bằng ros2 param set)
        # ---------------------------
        # topic nguồn ảnh (camera)
        self.declare_parameter("camera_topic", "/camera/image_raw")
        # topic nhận detections từ Perception Node
        self.declare_parameter("detection_topic", "/detections")
        # topic publish ảnh đã annotate (raw Image)
        self.declare_parameter("annotated_topic", "/camera/ai_annotated")
        # topic publish ảnh đã nén (CompressedImage) để stream
        self.declare_parameter(
            "annotated_compressed_topic", "/camera/ai_annotated/compressed"
        )
        # JPEG quality cho ảnh nén (0..100)
        self.declare_parameter("jpeg_quality", 80)
        # font scale cho label
        self.declare_parameter("font_scale", 0.6)
        # độ dày khung bbox
        self.declare_parameter("box_thickness", 2)
        # hiển thị confidence không
        self.declare_parameter("show_confidence", True)

        # Lấy giá trị param
        self.camera_topic = self.get_parameter("camera_topic").value
        self.detection_topic = self.get_parameter("detection_topic").value
        self.annotated_topic = self.get_parameter("annotated_topic").value
        self.annotated_compressed_topic = self.get_parameter(
            "annotated_compressed_topic"
        ).value
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.font_scale = float(self.get_parameter("font_scale").value)
        self.box_thickness = int(self.get_parameter("box_thickness").value)
        self.show_confidence = bool(self.get_parameter("show_confidence").value)

        # ---------------------------
        # CV bridge để chuyển đổi ROS Image <-> OpenCV
        # ---------------------------
        self.bridge = CvBridge()

        # ---------------------------
        # Publisher: ảnh annotate (Image) và ảnh nén (CompressedImage)
        # ---------------------------
        qos = QoSProfile(depth=5)
        self.annotated_pub = self.create_publisher(Image, self.annotated_topic, qos)
        self.annotated_compressed_pub = self.create_publisher(
            CompressedImage, self.annotated_compressed_topic, qos
        )

        # ---------------------------
        # Subscribers: ảnh camera và detections
        # - dùng callback song song; detections được lưu latest và dùng khi có ảnh
        # ---------------------------
        # Sub image: chúng ta xử lý mỗi frame ảnh đến
        self.create_subscription(Image, self.camera_topic, self.image_callback, qos)

        # Sub detections: lưu latest detections (thread-safe)
        self.latest_detections = None
        self._detections_lock = threading.Lock()
        self.create_subscription(
            Detection2DArray, self.detection_topic, self.detections_callback, qos
        )

        # Logger
        self.get_logger().info("AI Visualization Node started")
        self.get_logger().info(
            f"Camera topic: {self.camera_topic}, Detection topic: {self.detection_topic}"
        )
        self.get_logger().info(
            f"Publishing annotated: {self.annotated_topic} and {self.annotated_compressed_topic}"
        )

    # ---------------------------
    # Callback detections: lưu lại detections mới nhất (thread-safe)
    # ---------------------------
    def detections_callback(self, msg: Detection2DArray):
        """
        Mỗi khi PerceptionNode publish /detections, callback này được gọi.
        Ta chỉ lưu lại detections mới nhất để dùng khi có ảnh tiếp theo.
        """
        try:
            with self._detections_lock:
                # lưu object (bản sao reference) — nhẹ và đủ dùng
                self.latest_detections = msg
        except Exception as e:
            self.get_logger().error(f"Error in detections_callback: {e}")

    # ---------------------------
    # Callback ảnh camera: vẽ bounding box và publish
    # ---------------------------
    def image_callback(self, img_msg: Image):
        """
        Mỗi khi nhận frame ảnh:
        - chuyển ROS Image -> OpenCV
        - lấy latest detections (nếu có)
        - vẽ bbox + label
        - publish Image đã annotate và CompressedImage (JPEG)
        """
        try:
            # 1) Convert ROS Image -> OpenCV BGR
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        # 2) Copy ảnh để vẽ (tránh thay đổi buffer gốc)
        annotated = cv_image.copy()

        # 3) Lấy detections hiện tại một cách thread-safe
        with self._detections_lock:
            detections_msg = self.latest_detections

        # 4) Nếu có detections thì vẽ
        if detections_msg is not None and len(detections_msg.detections) > 0:
            # for each detection in Detection2DArray
            for det in detections_msg.detections:
                try:
                    # detection.bbox contains center (x,y) and size (size_x, size_y)
                    cx = det.bbox.center.position.x
                    cy = det.bbox.center.position.y
                    w = det.bbox.size_x
                    h = det.bbox.size_y

                    # Convert center/size -> pixel bounding box (left, top, right, bottom)
                    # NOTE: PerceptionNode của bạn đã ghi bbox theo pixel coordinate trong image frame.
                    left = int(cx - w / 2.0)
                    top = int(cy - h / 2.0)
                    right = int(cx + w / 2.0)
                    bottom = int(cy + h / 2.0)

                    # Clamp to image boundaries để tránh lỗi
                    left = max(0, min(left, annotated.shape[1] - 1))
                    right = max(0, min(right, annotated.shape[1] - 1))
                    top = max(0, min(top, annotated.shape[0] - 1))
                    bottom = max(0, min(bottom, annotated.shape[0] - 1))

                    # Lấy class name / score nếu có
                    label = ""
                    score_str = ""
                    if det.results and len(det.results) > 0:
                        # results[0].hypothesis.id chứa class name (PerceptionNode đặt là tên lớp)
                        try:
                            label = det.results[0].hypothesis.id
                        except Exception:
                            label = str(det.results[0].hypothesis.id)
                        # score nếu có
                        try:
                            score = det.results[0].hypothesis.score
                            if self.show_confidence:
                                score_str = f" {score:.2f}"
                        except Exception:
                            score_str = ""

                    # Text = "label 0.95" hoặc chỉ "label"
                    text = (
                        f"{label}{score_str}"
                        if label
                        else (score_str.strip() or "detected")
                    )

                    # Chọn màu theo lớp (hiện chỉ cố định màu đỏ cho person)
                    # Bạn có thể mở rộng mapping class->color nếu muốn
                    color = (0, 0, 255)  # BGR: đỏ

                    # Vẽ rectangle (bbox)
                    cv2.rectangle(
                        annotated,
                        (left, top),
                        (right, bottom),
                        color,
                        self.box_thickness,
                    )

                    # Prepare label background
                    ((text_w, text_h), baseline) = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
                    )
                    # location for label: trên góc trái bbox
                    label_x = left
                    label_y = top - 5
                    if label_y - text_h - baseline < 0:
                        # nếu không đủ chỗ trên đầu bbox thì vẽ phía dưới bbox
                        label_y = bottom + text_h + baseline + 5

                    # Rectangle background for text (filled)
                    cv2.rectangle(
                        annotated,
                        (label_x, label_y - text_h - baseline),
                        (label_x + text_w, label_y + 3),
                        color,
                        thickness=-1,
                    )

                    # Put text (white)
                    cv2.putText(
                        annotated,
                        text,
                        (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.font_scale,
                        (255, 255, 255),
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )

                except Exception as e:
                    # nếu một bounding box lỗi, log nhưng tiếp tục các box khác
                    self.get_logger().warn(f"Failed to draw one detection: {e}")
                    continue

        # 5) Publish annotated raw Image (sensor_msgs/Image)
        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            # giữ header tương tự ảnh gốc để dễ debug/đồng bộ (time stamp)
            annotated_msg.header = Header()
            annotated_msg.header.stamp = img_msg.header.stamp
            annotated_msg.header.frame_id = img_msg.header.frame_id
            self.annotated_pub.publish(annotated_msg)
        except CvBridgeError as e:
            self.get_logger().error(f"cv_bridge cv2_to_imgmsg failed: {e}")

        # 6) Publish CompressedImage (JPEG) cho stream
        try:
            # encode to JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            success, encimg = cv2.imencode(".jpg", annotated, encode_param)
            if success:
                comp_msg = CompressedImage()
                comp_msg.format = "jpeg"
                comp_msg.data = np.array(encimg).tobytes()
                # use same header timestamp/frame
                comp_msg.header = Header()
                comp_msg.header.stamp = img_msg.header.stamp
                comp_msg.header.frame_id = img_msg.header.frame_id
                self.annotated_compressed_pub.publish(comp_msg)
            else:
                self.get_logger().warn(
                    "JPEG encode failed, not publishing compressed image"
                )
        except Exception as e:
            self.get_logger().error(f"Failed to publish compressed image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = AIVisualizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("AI Visualization Node interrupted by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
