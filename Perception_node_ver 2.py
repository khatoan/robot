#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PerceptionNode_Lite (Python) for ROS2 Jazzy Jalisco
---------------------------------------------------
Nhiệm vụ:
- Điều khiển servo tilt (PCA9685/pigpio hoặc publish topic cho controller khác)
- Subscribe: /camera/image_raw, /lidar/scan, /imu/data
- Chạy YOLOv4-tiny (OpenCV DNN) để phát hiện người
- Publish:
    /detections (vision_msgs/Detection2DArray)
    /detection_markers (visualization_msgs/MarkerArray)
    /fusion_data_raw (std_msgs/String): chứa dữ liệu thô cho Fusion Node ghi CSV
- Không dựng PointCloud2 (giảm tải cho Pi 5)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image, LaserScan, Imu
from std_msgs.msg import Float32, String
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesis,
    ObjectHypothesisWithPose,
)
from visualization_msgs.msg import Marker, MarkerArray
import message_filters
import cv2
import numpy as np
from cv_bridge import CvBridge
import threading, time, math, os


# -------------------------------------------------------------
# Servo controller: hỗ trợ cả PCA9685, pigpio hoặc chỉ publish góc tilt
# -------------------------------------------------------------
class ServoController:
    def __init__(
        self,
        node: Node,
        topic_name="servo/tilt_angle",
        mode="topic",
        pin=18,
        min_us=500,
        max_us=2500,
    ):
        self.node = node
        self.mode = mode
        self.pin = pin
        self.min_us = min_us
        self.max_us = max_us
        self._angle = 0.0
        self._lock = threading.Lock()
        self.angle_pub = node.create_publisher(Float32, topic_name, 10)

        # Nếu không có phần cứng PWM, chỉ publish topic
        if self.mode == "hardware":
            try:
                from adafruit_pca9685 import PCA9685
                import board, busio

                i2c = busio.I2C(board.SCL, board.SDA)
                self._pca = PCA9685(i2c)
                self._pca.frequency = 50
                node.get_logger().info("ServoController: PCA9685 sẵn sàng.")
            except Exception as e:
                node.get_logger().warn(
                    f"Không tìm thấy PCA9685 ({e}), fallback sang topic mode."
                )
                self.mode = "topic"

    def set_angle(self, angle_deg: float):
        with self._lock:
            self._angle = float(angle_deg)
        msg = Float32()
        msg.data = float(angle_deg)
        self.angle_pub.publish(msg)

    def get_angle(self):
        with self._lock:
            return self._angle


# -------------------------------------------------------------
# Perception Node chính
# -------------------------------------------------------------
class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node_lite")

        # Tham số cơ bản
        self.declare_parameter("model_cfg", "config/yolov4-tiny.cfg")
        self.declare_parameter("model_weights", "config/yolov4-tiny.weights")
        self.declare_parameter("names_file", "config/coco.names")
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("nms_threshold", 0.4)
        self.declare_parameter("frame_skip", 0)
        self.declare_parameter("servo_mode", "topic")
        self.declare_parameter("servo_sweep_min", 0.0)
        self.declare_parameter("servo_sweep_max", 90.0)
        self.declare_parameter("servo_sweep_step", 2.0)
        self.declare_parameter("servo_sweep_rate_hz", 10.0)

        # Topics
        self.declare_parameter("camera_topic", "/camera/image_raw")
        self.declare_parameter("lidar_topic", "/lidar/scan")
        self.declare_parameter("imu_topic", "/imu/data")
        self.declare_parameter("fusion_topic", "/fusion_data_raw")

        # Đọc tham số
        self.camera_topic = self.get_parameter("camera_topic").value
        self.lidar_topic = self.get_parameter("lidar_topic").value
        self.imu_topic = self.get_parameter("imu_topic").value
        self.fusion_topic = self.get_parameter("fusion_topic").value

        # Load YOLO
        self._load_yolo_model()
        self.bridge = CvBridge()

        # Servo
        self.servo = ServoController(
            self,
            topic_name="/lidar/tilt_angle",
            mode=self.get_parameter("servo_mode").value,
        )
        self._tilt_lock = threading.Lock()
        self._current_tilt = self.servo.get_angle()

        # Publisher
        qos = QoSProfile(depth=10)
        self.detection_pub = self.create_publisher(Detection2DArray, "/detections", qos)
        self.marker_pub = self.create_publisher(MarkerArray, "/detection_markers", qos)
        self.raw_pub = self.create_publisher(String, self.fusion_topic, qos)

        # IMU lưu yaw
        self._latest_yaw = 0.0
        self._imu_lock = threading.Lock()

        # Subscriptions đồng bộ camera + lidar + imu
        camera_sub = message_filters.Subscriber(self, Image, self.camera_topic)
        lidar_sub = message_filters.Subscriber(self, LaserScan, self.lidar_topic)
        imu_sub = message_filters.Subscriber(self, Imu, self.imu_topic)

        sync = message_filters.ApproximateTimeSynchronizer(
            [camera_sub, lidar_sub, imu_sub], 10, 0.1
        )
        sync.registerCallback(self.sync_callback)

        # Thread quét servo
        threading.Thread(target=self._servo_sweep_loop, daemon=True).start()

        self.get_logger().info("✅ PerceptionNode_Lite khởi động thành công.")

    # -------------------------------------------------------------
    # Load YOLO model
    # -------------------------------------------------------------
    def _load_yolo_model(self):
        try:
            cfg = os.path.expanduser(self.get_parameter("model_cfg").value)
            weights = os.path.expanduser(self.get_parameter("model_weights").value)
            names = os.path.expanduser(self.get_parameter("names_file").value)
            self.net = cv2.dnn.readNetFromDarknet(cfg, weights)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            with open(names) as f:
                self.class_names = [l.strip() for l in f.readlines()]
            self.get_logger().info("YOLO model loaded thành công.")
        except Exception as e:
            self.get_logger().error(f"Lỗi load YOLO: {e}")
            raise

    # -------------------------------------------------------------
    # IMU callback → cập nhật góc yaw ψ (radian)
    # -------------------------------------------------------------
    def imu_callback(self, imu_msg: Imu):
        q = imu_msg.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        with self._imu_lock:
            self._latest_yaw = yaw

    # -------------------------------------------------------------
    # Servo sweep (lắc nghiêng lidar)
    # -------------------------------------------------------------
    def _servo_sweep_loop(self):
        try:
            min_a = self.get_parameter("servo_sweep_min").value
            max_a = self.get_parameter("servo_sweep_max").value
            step = self.get_parameter("servo_sweep_step").value
            rate = self.get_parameter("servo_sweep_rate_hz").value
            period = 1.0 / rate
            angle = min_a
            direction = 1
            while rclpy.ok():
                self.servo.set_angle(angle)
                with self._tilt_lock:
                    self._current_tilt = angle
                angle += direction * step
                if angle >= max_a or angle <= min_a:
                    direction *= -1
                time.sleep(period)
        except Exception as e:
            self.get_logger().error(f"Lỗi servo sweep: {e}")

    # -------------------------------------------------------------
    # Callback chính: đồng bộ Camera + Lidar + IMU
    # -------------------------------------------------------------
    def sync_callback(self, camera_msg, lidar_msg, imu_msg):
        try:
            # Convert ảnh
            frame = self.bridge.imgmsg_to_cv2(camera_msg, "bgr8")

            # Chạy YOLO
            detections = self._run_yolo(frame)

            # Publish kết quả phát hiện người
            det_array = Detection2DArray()
            det_array.header = camera_msg.header
            marker_array = MarkerArray()
            mid = 0
            for cid, cname, conf, box in detections:
                if cname != "person":
                    continue
                det = Detection2D()
                left, top, w, h = box
                det.bbox.center.position.x = left + w / 2
                det.bbox.center.position.y = top + h / 2
                det.bbox.size_x = w
                det.bbox.size_y = h
                oh = ObjectHypothesis()
                oh.id, oh.score = cname, conf
                ohwp = ObjectHypothesisWithPose()
                ohwp.hypothesis = oh
                det.results = [ohwp]
                det_array.detections.append(det)
                # RViz marker (đơn giản)
                m = Marker()
                m.header = camera_msg.header
                m.id = mid
                m.type = Marker.CUBE
                m.scale.x, m.scale.y, m.scale.z = 0.1, 0.1, 0.05
                m.color.a, m.color.r = 0.6, 1.0
                marker_array.markers.append(m)
                mid += 1
            self.detection_pub.publish(det_array)
            self.marker_pub.publish(marker_array)

            # Gửi dữ liệu thô cho Fusion Node
            with self._tilt_lock:
                phi = self._current_tilt
            with self._imu_lock:
                psi = self._latest_yaw
            timestamp = (
                camera_msg.header.stamp.sec + camera_msg.header.stamp.nanosec * 1e-9
            )
            # Gói dữ liệu: timestamp, phi_deg, ranges..., psi
            raw_str = f"{timestamp},{phi:.2f},{psi:.4f}," + ",".join(
                [f"{r:.3f}" for r in lidar_msg.ranges]
            )
            msg = String()
            msg.data = raw_str
            self.raw_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f"Lỗi sync_callback: {e}")

    # -------------------------------------------------------------
    # YOLO inference
    # -------------------------------------------------------------
    def _run_yolo(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame_bgr, 1 / 255.0, (416, 416), swapRB=True, crop=False
        )
        self.net.setInput(blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        class_ids, confidences, boxes = [], [], []
        for out in outs:
            for i in range(out.shape[0]):
                scores = out[i][5:]
                cid = np.argmax(scores)
                conf = scores[cid] * out[i][4]
                if conf > self.get_parameter("confidence_threshold").value:
                    cx, cy, bw, bh = (
                        out[i][0] * w,
                        out[i][1] * h,
                        out[i][2] * w,
                        out[i][3] * h,
                    )
                    x, y = int(cx - bw / 2), int(cy - bh / 2)
                    boxes.append([x, y, int(bw), int(bh)])
                    confidences.append(float(conf))
                    class_ids.append(int(cid))
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        detections = []
        for i in np.array(indices).flatten():
            cname = (
                self.class_names[class_ids[i]]
                if class_ids[i] < len(self.class_names)
                else str(class_ids[i])
            )
            detections.append((class_ids[i], cname, confidences[i], boxes[i]))
        return detections


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Dừng PerceptionNode.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
