# robot/fusion_node.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fusion Node
- Subscribe:
    - /lidar/scan (sensor_msgs/LaserScan)
    - /lidar/tilt_angle (std_msgs/Float32)
    - /pose (geometry_msgs/Pose2D)
- When all three available for a frame -> write CSV row for WebApp 3D mapping
CSV format:
timestamp, phi_deg, theta0_deg, r0_m, theta1_deg, r1_m, ..., x_r, y_r, psi_r
"""
# Import các thư viện cần thiết
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from geometry_msgs.msg import Pose2D
import csv, os, math
from datetime import datetime, timezone
from rclpy.qos import qos_profile_sensor_data


class FusionNode(Node):

    # Khởi tạo FusionNode
    def __init__(self):

        # Khởi tạo ROS2 node và đọc tham số cấu hình đường dẫn file CSV đầu ra
        super().__init__("fusion_node")
        # self.declare_parameter("output_csv", "lidar_imu_odom_log.csv")
        # self.csv_path = self.get_parameter("output_csv").value
        self.declare_parameter("output_dir", "/tmp")
        self.output_dir = self.get_parameter("output_dir").value
        if not os.path.isdir(self.output_dir):
            self.get_logger().warn(
                f"Output dir '{self.output_dir}' not found, fallback to /tmp"
            )
            self.output_dir = "/tmp"
        os.makedirs(self.output_dir, exist_ok=True)

        # tạo tên file theo timestamp để tránh ghi đè
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(self.output_dir, f"fusion_log_{ts}.csv")

        self.header_written = False

        self.get_logger().info(f"FusionNode started, CSV output: {self.csv_path}")

        # Lưu dữ liệu mới nhất từ các nguồn bất đồng bộ, dùng để ghép thành một frame fusion hoàn chỉnh
        self.latest_lidar = None
        self.latest_phi = None
        self.latest_pose = None

        # Đăng ký các subscriber
        self.create_subscription(
            LaserScan, "/scan", self.lidar_cb, qos_profile_sensor_data
        )
        self.create_subscription(Float32, "/lidar/tilt_angle", self.tilt_cb, 20)
        self.create_subscription(Pose2D, "/pose", self.pose_cb, 20)

        # Kiểm tra file CSV đã tồn tại để tránh ghi trùng header, đảm bảo thư mục lưu file tồn tại và log trạng thái khởi động node
        # self.header_written = os.path.exists(self.csv_path)
        # os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        # self.get_logger().info("FusionNode started, writing to: " + self.csv_path)

    # Hàm nhận dữ iệu LIDAR và cố gắng ghi dữ liệu fusion nếu có đủ thông tin
    def lidar_cb(self, msg: LaserScan):
        self.latest_lidar = msg
        self.try_write()

    # Hàm nhận dữ liệu góc nghiêng và cố gắng ghi dữ liệu fusion nếu có đủ thông tin
    def tilt_cb(self, msg: Float32):
        self.latest_phi = float(msg.data)
        self.try_write()

    # Hàm nhận dữ liệu vị trí và cố gắng ghi dữ liệu fusion nếu có đủ thông tin
    def pose_cb(self, msg: Pose2D):
        self.latest_pose = msg
        self.try_write()

    # Hàm chuẩn hóa góc yaw về khoảng [-180, 180] độ
    def _normalize_yaw(self, yaw_deg):
        yaw_deg = yaw_deg % 360
        if yaw_deg > 180:
            yaw_deg -= 360
        return yaw_deg

    # Hàm ghi dữ liệu fusion vào file CSV nếu có đủ thông tin từ LIDAR, góc nghiêng và vị trí
    def try_write(self):
        # Kiểm tra đủ dữ liệu để ghi
        if not (
            self.latest_lidar and (self.latest_phi is not None) and self.latest_pose
        ):
            return
        lidar = self.latest_lidar
        phi = float(self.latest_phi)
        pose = self.latest_pose
        psi_deg = self._normalize_yaw(math.degrees(pose.theta))
        timestamp_str = (
            datetime.now(timezone.utc).isoformat(timespec="milliseconds") + "Z"
        )
        arr = []
        angle = lidar.angle_min
        for r in lidar.ranges:
            arr.append(math.degrees(angle))
            arr.append(float(r))
            angle += lidar.angle_increment
        row = (
            [timestamp_str, float(phi)]
            + arr
            + [float(pose.x), float(pose.y), float(psi_deg)]
        )
        write_header = not self.header_written
        # Ghi dòng dữ liệu vào file CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                num_beams = len(lidar.ranges)
                header = ["timestamp", "phi_deg"]
                for i in range(num_beams):
                    header.append(f"theta{i}_deg")
                    header.append(f"r{i}_m")
                header += ["x_r", "y_r", "psi_r"]
                writer.writerow(header)
                self.header_written = True
                self.get_logger().info(
                    f"CSV initialized: {num_beams} beams -> {self.csv_path}"
                )
            writer.writerow(row)
        self.get_logger().info(
            f"wrote frame: {len(lidar.ranges)} points, phi={phi:.1f}°, psi={psi_deg:.1f}°"
        )
        # Xóa dữ liệu đã ghi để chờ dữ liệu mới
        self.latest_lidar = None


def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
