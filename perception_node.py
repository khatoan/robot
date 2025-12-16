# robot/perception_node.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import các thư viện cần thiết
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import threading, time, math, os, io, base64, traceback
from sensor_msgs.msg import Image, LaserScan, Imu
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Float32, String
import serial

# Thử import pigpio (chỉ có trên Raspberry Pi). Nếu không có → gán None để dùng chế độ fallback (không điều khiển servo qua GPIO).
try:
    import lgpio
except Exception:
    lgpio = None

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
        pin=18,
        min_us=500,
        max_us=2500,
        freq_hz=50,
    ):
        self.node = node
        self.mode = mode
        self.pin = pin
        self.min_us = min_us
        self.max_us = max_us
        self.freq = freq_hz

        self._angle = 0.0
        self._lock = threading.Lock()

        self.pub = node.create_publisher(Float32, topic_name, 10)

        # lgpio state
        self._lgpio_handle = None
        self._period_us = int(1_000_000 / self.freq)

        # ===== GPIO init =====
        if self.mode == "gpio":
            if lgpio is None:
                node.get_logger().warning("lgpio not available; fallback to topic mode")
                self.mode = "topic"
            else:
                try:
                    self._lgpio_handle = lgpio.gpiochip_open(4)
                    # lgpio.gpio_claim_output(self._lgpio_handle, self.pin)
                    node.get_logger().info(
                        f"ServoController: lgpio ready on GPIO {self.pin}"
                    )
                except Exception as e:
                    node.get_logger().warning(
                        f"lgpio init error: {e}; fallback to topic mode"
                    )
                    self.mode = "topic"
                    self._lgpio_handle = None

    def _angle_to_pulse(self, angle_deg: float) -> int:
        angle_deg = max(0.0, min(180.0, angle_deg))
        return int(self.min_us + (angle_deg / 180.0) * (self.max_us - self.min_us))

    def set_angle(self, angle_deg: float):
        with self._lock:
            self._angle = float(angle_deg)

        if self.mode == "gpio" and self._lgpio_handle is not None:
            try:
                pulse_us = self._angle_to_pulse(angle_deg)
                duty = (pulse_us / self._period_us) * 100.0

                lgpio.tx_pwm(
                    self._lgpio_handle,
                    self.pin,
                    self.freq,
                    duty,
                )
            except Exception as e:
                self.node.get_logger().warning(
                    f"lgpio pwm error: {e}; fall back to topic mode"
                )
                self.mode = "topic"

        msg = Float32()
        msg.data = float(angle_deg)
        self.pub.publish(msg)

    def get_angle(self):
        with self._lock:
            return self._angle

    def shutdown(self):
        if self._lgpio_handle is not None:
            try:
                lgpio.tx_pwm(self._lgpio_handle, self.pin, 0, 0)
                lgpio.gpiochip_close(self._lgpio_handle)
            except Exception:
                pass


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
        self.declare_parameter("imu_topic", "/imu/data")
        self.declare_parameter("servo_mode", "gpio")
        self.declare_parameter("servo_pin", 18)
        self.declare_parameter("servo_min_us", 500)
        self.declare_parameter("servo_max_us", 2500)
        self.declare_parameter("servo_sweep_min", 0.0)
        self.declare_parameter("servo_sweep_max", 90.0)
        self.declare_parameter("servo_sweep_step", 2.0)
        self.declare_parameter("nms_threshold", 0.4)
        self.declare_parameter("servo_sweep_rate_hz", 10.0)

        # Lấy giá trị tham số
        self.imu_topic = self.get_parameter("imu_topic").value

        # publishers
        qos = QoSProfile(depth=10)
        self.pose_pub = self.create_publisher(Pose2D, "/pose", qos)
        self.tilt_pub = self.create_publisher(Float32, "/lidar/tilt_angle", qos)
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
        # Nếu MPU6050 có sẵn: đọc rate z và tích hợp để cập nhật yaw.
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
            if hasattr(node.servo, "shutdown"):
                node.servo.shutdown()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()
