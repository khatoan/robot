# robot/perception_node.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perception Node (ROS2, Python)
- Điều khiển servo tilt qua pigpio (GPIO) hoặc publish topic khi pigpio không khả dụng
- Subscribe camera IMX708 (topic '/camera/imx708/image_raw') và IMX219 ('/camera/imx219/image_raw')
- Subscribe LIDAR driver topic '/lidar/scan' (sensor_msgs/LaserScan)
- Đọc IMU MPU6050 qua I2C nếu không có topic '/imu/data'
- Chạy YOLOv4-tiny (OpenCV DNN) để detect người
- Publish:
    - /detections (vision_msgs/Detection2DArray)
    - /detection_markers (visualization_msgs/MarkerArray)
    - /pose (geometry_msgs/Pose2D)
    - /lidar/tilt_angle (std_msgs/Float32)
    - /camera/image_web (sensor_msgs/Image)  -- ảnh nén/jpg để web gửi
"""

# Import các thư viện cần thiết
# ROS2 core + tiện ích Python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import threading, time, math, os, io, base64, traceback

# ROS2 message types (ảnh, lidar, imu, pose, detection, marker
from sensor_msgs.msg import Image, LaserScan, Imu
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Float32, String
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesis
from visualization_msgs.msg import Marker, MarkerArray

# Xử lý ảnh (ROS Image ↔ OpenCV) + YOLO
from cv_bridge import CvBridge
import cv2
import numpy as np

# Thử import pigpio (chỉ có trên Raspberry Pi). Nếu không có → gán None để dùng chế độ fallback (không điều khiển servo qua GPIO).
try:
    import pigpio
except Exception:
    pigpio = None

# Thử import smbus2 để đọc MPU6050 qua I2C. Nếu không có → gán None để vô hiệu hóa đọc trực tiếp.
try:
    from smbus2 import SMBus
except Exception:
    SMBus = None

# Tiện ích ROS2 để lấy đường dẫn package
from ament_index_python.packages import get_package_share_directory


class ServoController:
    def __init__(
        self,
        node: Node,
        topic_name="/lidar/tilt_angle",
        mode="gpio",
        pin=18,  # Chân GPIO cho servo
        min_us=500,  # Xung cho góc 0 độ
        max_us=2500,  # Xung cho góc 180 độ
    ):
        # Lưu cấu hình servo và trạng thái góc hiện tại
        self.node = node
        self.mode = mode
        self.pin = pin
        self.min_us = min_us
        self.max_us = max_us
        self._angle = 0.0
        self._lock = threading.Lock()
        # Tạo publisher cho topic góc tilt
        self.pub = node.create_publisher(Float32, topic_name, 10)
        self.pi = None

        # Kiểm tra và khởi tạo pigpio nếu ở chế độ gpio
        if self.mode == "gpio":
            # Kiểm tra pigpio có sẵn không
            if pigpio is None:
                node.get_logger().warning(
                    "pigpio not installed; fallback to topic mode"
                )
                # Chuyển sang chế độ topic nếu không có pigpio
                self.mode = "topic"
            else:
                try:
                    self.pi = pigpio.pi()
                    if not self.pi.connected:
                        node.get_logger().warning(
                            "pigpiod not connected; fallback to topic mode"
                        )
                        self.mode = "topic"
                    else:
                        self.node.get_logger().info(
                            f"ServoController: pigpio ready on pin {pin}"
                        )
                except Exception as e:
                    node.get_logger().warning(
                        f"pigpio init error: {e}; fallback to topic mode"
                    )
                    self.mode = "topic"

    # Set góc tilt: điều khiển servo (nếu có) và publish góc qua topic
    def set_angle(self, angle_deg: float):
        with self._lock:
            self._angle = float(angle_deg)
        pulse = int(self.min_us + (angle_deg / 180.0) * (self.max_us - self.min_us))
        if self.mode == "gpio" and self.pi:
            try:
                self.pi.set_servo_pulsewidth(self.pin, pulse)
            except Exception as e:
                self.node.get_logger().warning(f"Failed set servo pwm: {e}")
        msg = Float32()
        msg.data = float(angle_deg)
        self.pub.publish(msg)

    # Lấy góc tilt hiện tại
    def get_angle(self):
        with self._lock:
            return self._angle


class MPU6050Reader:
    """Đọc MPU6050 qua I2C (smbus2). Nếu không có smbus, raise"""

    # Khởi tạo MPU6050 qua I2C; nếu không khả dụng thì tự động vô hiệu hóa
    def __init__(self, node: Node, i2c_bus=1, address=0x68):
        self.node = node
        self.addr = address
        self.bus = None
        self.available = False
        if SMBus is None:
            node.get_logger().warning(
                "smbus2 not available; MPU6050 direct read disabled"
            )
            return
        try:
            self.bus = SMBus(i2c_bus)
            # Wake up MPU6050 (power management)
            self.bus.write_byte_data(self.addr, 0x6B, 0)
            self.available = True
            node.get_logger().info("MPU6050 reader initialised")
        except Exception as e:
            node.get_logger().warning(f"MPU6050 init failed: {e}")
            self.available = False

    # Đọc gyro Z (MPU6050), chuyển sang rad/s; yaw được tích hợp ở bên ngoài
    def read_yaw(self):
        if not self.available:
            raise RuntimeError("MPU6050 not available")
        try:
            # gyro Z register 0x47..0x48 (16-bit)
            gz_h = self.bus.read_byte_data(self.addr, 0x47)
            gz_l = self.bus.read_byte_data(self.addr, 0x48)
            gz = (gz_h << 8) | gz_l
            if gz & 0x8000:
                gz = -((~gz & 0xFFFF) + 1)
            # cảm biến trả giá trị in deg/sec / 131 (ở +-250deg/s)
            rate_dps = gz / 131.0
            # convert to rad/s
            rate_rps = math.radians(rate_dps)
            # trả về yaw rate; góc yaw được tính bên ngoài bằng rate * dt
            return rate_rps
        except Exception as e:
            self.node.get_logger().warning(f"MPU read error: {e}")
            raise


class PerceptionNode(Node):
    def __init__(self):
        # Khởi tạo ROS2 node với tên perception_node
        super().__init__("perception_node")

        # Khai báo các tham số cấu hình node
        self.declare_parameter("camera_imx708_topic", "/camera/imx708/image_raw")
        self.declare_parameter("camera_imx219_topic", "/camera/imx219/image_raw")
        self.declare_parameter("lidar_topic", "/lidar/scan")
        self.declare_parameter("imu_topic", "/imu/data")
        self.declare_parameter("servo_mode", "gpio")
        self.declare_parameter("servo_pin", 18)
        self.declare_parameter("servo_min_us", 500)
        self.declare_parameter("servo_max_us", 2500)
        self.declare_parameter("servo_sweep_min", 0.0)
        self.declare_parameter("servo_sweep_max", 90.0)
        self.declare_parameter("servo_sweep_step", 2.0)
        self.declare_parameter("frame_skip", 0)
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("nms_threshold", 0.4)
        self.declare_parameter("model_cfg", "yolov4-tiny.cfg")
        self.declare_parameter("model_weights", "yolov4-tiny.weights")
        self.declare_parameter("names_file", "coco.names")

        # Lấy giá trị tham số
        self.cam1_topic = self.get_parameter("camera_imx708_topic").value
        self.cam2_topic = self.get_parameter("camera_imx219_topic").value
        self.lidar_topic = self.get_parameter("lidar_topic").value
        self.imu_topic = self.get_parameter("imu_topic").value

        # CvBridge để chuyển đổi ROS Image ↔ OpenCV
        self.bridge = CvBridge()

        # publishers
        qos = QoSProfile(depth=10)
        self.detections_pub = self.create_publisher(
            Detection2DArray, "/detections", qos
        )
        self.marker_pub = self.create_publisher(MarkerArray, "/detection_markers", qos)
        self.pose_pub = self.create_publisher(Pose2D, "/pose", qos)
        self.tilt_pub = self.create_publisher(Float32, "/lidar/tilt_angle", qos)
        self.image_web_pub = self.create_publisher(Image, "/camera/image_web", qos)

        # Khởi tạo ServoController
        servo_mode = self.get_parameter("servo_mode").value
        pin = self.get_parameter("servo_pin").value
        min_us = self.get_parameter("servo_min_us").value
        max_us = self.get_parameter("servo_max_us").value
        self.servo = ServoController(
            self,
            topic_name="/lidar/tilt_angle",
            mode=servo_mode,
            pin=pin,
            min_us=min_us,
            max_us=max_us,
        )
        self._tilt_lock = threading.Lock()
        self._current_tilt = self.servo.get_angle()

        # IMU: ưu tiên đọc MPU6050 trực tiếp hoặc nhận dữ liệu từ topic /imu/data nếu không có phần cứng
        self._mpu = MPU6050Reader(self)
        self._latest_imu = None
        self._imu_lock = threading.Lock()

        # Trạng thái pose nội bộ (dead-reckoning) và lock bảo vệ dữ liệu
        self._pose_lock = threading.Lock()
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0  # rad
        self._last_time = (
            self.get_clock().now().to_msg().sec
            + self.get_clock().now().to_msg().nanosec * 1e-9
        )

        # Load model YOLOv4-tiny
        self._load_yolo_model()

        # Subscribe các topic camera, IMU và lidar
        self.create_subscription(Image, self.cam1_topic, self.camera_callback, 10)
        self.create_subscription(Image, self.cam2_topic, self.camera_callback, 10)
        self.create_subscription(LaserScan, self.lidar_topic, self.lidar_callback, 10)
        # Subscribe topic imu nếu có
        try:
            self.create_subscription(Imu, self.imu_topic, self.imu_callback, 20)
        except Exception:
            self.get_logger().warning(
                "Không thể subscribe imu topic; sẽ dùng MPU reader nếu có."
            )

        # Chạy song song các tác vụ nền, tránh block ROS callbacks
        threading.Thread(target=self._servo_sweep_loop, daemon=True).start()
        if self._mpu.available:
            threading.Thread(target=self._mpu_integration_loop, daemon=True).start()

        # Thông báo khởi động thành công
        self.get_logger().info("PerceptionNode khởi động thành công.")

    # Load mô hình YOLOv4-tiny từ thư mục config của ROS package
    def _load_yolo_model(self):
        try:
            pkg_share = get_package_share_directory("robot")
            cfg = os.path.join(
                pkg_share, "config", self.get_parameter("model_cfg").value
            )
            weights = os.path.join(
                pkg_share, "config", self.get_parameter("model_weights").value
            )
            names = os.path.join(
                pkg_share, "config", self.get_parameter("names_file").value
            )
            if not all([os.path.exists(p) for p in (cfg, weights, names)]):
                raise FileNotFoundError(
                    "YOLO config/weights/names not found under package config/"
                )
            self.net = cv2.dnn.readNet(weights, cfg)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            with open(names) as f:
                self.class_names = [l.strip() for l in f.readlines()]
            self.get_logger().info("YOLOv4-tiny loaded.")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model: {e}")
            raise

    # ---------- callbacks ----------
    # Xử lý frame từ camera
    def camera_callback(self, msg: Image):
        """Xử lý frame từ camera (cả IMX708 & IMX219)
        - convert -> run YOLO -> publish detections -> publish image_web (jpeg)
        """
        try:
            # Chuyển Image ROS sang OpenCV (BGR)
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Chạy YOLO để lấy danh sách detection
            detections = self._run_yolo(frame)

            # Chuẩn bị message detection và marker
            det_array = Detection2DArray()
            det_array.header = msg.header
            marker_array = MarkerArray()
            mid = 0
            for cid, cname, conf, box in detections:
                # Chỉ quan tâm đến class "person"
                if cname != "person":
                    continue
                left, top, w, h = box

                # Chuyển bounding box YOLO sang Detection2D
                det = Detection2D()
                det.bbox.center.position.x = float(left + w / 2)
                det.bbox.center.position.y = float(top + h / 2)
                det.bbox.size_x = float(w)
                det.bbox.size_y = float(h)
                oh = ObjectHypothesis()
                oh.id = cname
                oh.score = float(conf)
                det.results = [oh]
                det_array.detections.append(det)

                # Tạo marker hình hộp cho detection
                m = Marker()
                m.header = msg.header
                m.id = mid
                m.type = Marker.CUBE
                m.scale.x, m.scale.y, m.scale.z = 0.1, 0.1, 0.05
                m.color.a, m.color.r = 0.6, 1.0
                marker_array.markers.append(m)
                mid += 1

            # Publish kết quả detections và markers
            self.detections_pub.publish(det_array)
            self.marker_pub.publish(marker_array)

            # Vẽ bounding box YOLO lên ảnh trước khi stream web (thêm sau này)
            for cid, cname, conf, box in detections:
                if cname != "person":
                    continue

                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{cname} {conf:.2f}",
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

            # Chuẩn bị và publish ảnh nén JPEG cho web
            ret, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            # Nếu nén thành công
            if ret:
                img_msg = Image()
                img_msg.header = msg.header
                img_msg.height = frame.shape[0]
                img_msg.width = frame.shape[1]
                img_msg.encoding = "jpeg"
                img_msg.is_bigendian = 0
                img_msg.step = len(jpg.tobytes())
                img_msg.data = jpg.tobytes()
                # Publish ảnh nén
                self.image_web_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().error(
                f"camera_callback error: {e}\n{traceback.format_exc()}"
            )

    # Callback LIDAR (hiện chưa xử lý, dành cho mở rộng sau)
    def lidar_callback(self, msg: LaserScan):
        pass

    # Cập nhật yaw từ topic IMU và publish Pose2D (nếu có bên publish topic imu)
    def imu_callback(self, msg: Imu):
        try:
            q = msg.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            with self._imu_lock:
                self._latest_imu = yaw
            # Publish yaw; vị trí x,y mặc định = 0 nếu chưa có odom
            with self._pose_lock:
                self._yaw = yaw
            pose = Pose2D()
            pose.x = float(self._x)
            pose.y = float(self._y)
            pose.theta = float(self._yaw)
            self.pose_pub.publish(pose)
        except Exception as e:
            self.get_logger().warning(f"imu_callback error: {e}")

    # Vòng lặp quét servo tilt
    # Chỉ quan tâm logic góc quét và publish góc tilt gián tiếp qua ServoController
    def _servo_sweep_loop(self):
        """Vòng lặp scan servo tilt: sweep giữa min->max theo step và publish tilt topic"""
        try:
            min_a = self.get_parameter("servo_sweep_min").value
            max_a = self.get_parameter("servo_sweep_max").value
            step = self.get_parameter("servo_sweep_step").value
            rate_hz = (
                float(
                    self.get_parameter("servo_sweep_rate_hz")
                    .get_parameter_value()
                    .double_value
                )
                if self.has_parameter("servo_sweep_rate_hz")
                else 10.0
            )
        except Exception:
            min_a, max_a, step, rate_hz = 0.0, 90.0, 2.0, 10.0
        period = 1.0 / max(rate_hz, 1.0)
        angle = min_a
        direction = 1
        while rclpy.ok():
            try:
                self.servo.set_angle(angle)
                with self._tilt_lock:
                    self._current_tilt = angle
                angle += direction * step
                if angle >= max_a or angle <= min_a:
                    direction *= -1
                time.sleep(period)
            except Exception as e:
                self.get_logger().error(f"servo loop error: {e}")
                time.sleep(0.5)

    # Vòng lặp tích hợp yaw từ MPU6050 nếu không có topic IMU
    def _mpu_integration_loop(self):
        """Nếu MPU6050 có sẵn: đọc rate z và tích hợp để cập nhật yaw.
        NOTE: đây là giải pháp đơn giản, drift sẽ xảy ra; dùng để có Theta nhanh khi không có odom.
        """
        last_t = time.time()
        while rclpy.ok():
            try:
                rate = self._mpu.read_yaw()  # rad/s (approx)
                now = time.time()
                dt = now - last_t
                last_t = now
                with self._pose_lock:
                    self._yaw += rate * dt
                # publish pose periodically
                pose = Pose2D()
                pose.x = float(self._x)
                pose.y = float(self._y)
                pose.theta = float(self._yaw)
                self.pose_pub.publish(pose)
                time.sleep(0.05)
            except Exception:
                time.sleep(0.1)

    # Chạy YOLOv4-tiny trên frame BGR và trả về danh sách detection
    def _run_yolo(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame_bgr, 1 / 255.0, (416, 416), swapRB=True, crop=False
        )
        self.net.setInput(blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        class_ids, confidences, boxes = [], [], []
        conf_th = float(self.get_parameter("confidence_threshold").value)
        for out in outs:
            for i in range(out.shape[0]):
                scores = out[i][5:]
                if scores.size == 0:
                    continue
                cid = int(np.argmax(scores))
                conf = float(scores[cid]) * float(out[i][4])
                if conf > conf_th:
                    cx, cy, bw, bh = (
                        out[i][0] * w,
                        out[i][1] * h,
                        out[i][2] * w,
                        out[i][3] * h,
                    )
                    x, y = int(cx - bw / 2), int(cy - bh / 2)
                    boxes.append([x, y, int(bw), int(bh)])
                    confidences.append(conf)
                    class_ids.append(cid)
        if len(boxes) == 0:
            return []
        nms = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            conf_th,
            float(self.get_parameter("nms_threshold").value),
        )
        dets = []
        if isinstance(nms, (list, tuple)) or nms.ndim == 1:
            indices = np.array(nms).flatten()
        else:
            indices = nms.flatten()
        for i in indices:
            cname = (
                self.class_names[class_ids[i]]
                if class_ids[i] < len(self.class_names)
                else str(class_ids[i])
            )
            dets.append((class_ids[i], cname, float(confidences[i]), boxes[i]))
        return dets


# Entry point: khởi chạy và shutdown PerceptionNode an toàn
def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("PerceptionNode dừng bởi người dùng")
    finally:
        try:
            node.servo.set_angle(0.0)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()
