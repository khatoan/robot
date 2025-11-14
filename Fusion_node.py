#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fusion Node – Ghi dữ liệu thô cho WebApp 3D Mapping
---------------------------------------------------
Node này có nhiệm vụ:
- Nhận Lidar + IMU + Odom + Tilt Angle
- Hợp nhất chúng lại
- Ghi 1 dòng CSV theo đúng định dạng WebApp yêu cầu:
  timestamp, phi_deg, theta0_deg, r0_m, theta1_deg, r1_m, ..., x_r, y_r, psi_r

Không xử lý bản đồ, không lọc dữ liệu, không pointcloud.
Chỉ ghi dữ liệu "thô nhưng sạch".

Dễ chạy trên Pi 5 vì rất nhẹ.
"""

import rclpy
from rclpy.node import Node
import csv
import os
import math
from datetime import datetime, timezone

# ROS msgs
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32


class FusionNode(Node):

    def __init__(self):
        super().__init__("fusion_node_py")

        # -----------------------------
        # Load parameters
        # -----------------------------
        self.declare_parameter("output_csv", "lidar_imu_odom_log.csv")
        self.csv_path = (
            self.get_parameter("output_csv").get_parameter_value().string_value
        )

        # -----------------------------
        # Internal buffer
        # -----------------------------
        self.latest_lidar = None
        self.latest_phi_deg = None
        self.latest_imu = None  # yaw in radians
        self.latest_odom = None

        # -----------------------------
        # State
        # -----------------------------
        self.header_written = os.path.exists(
            self.csv_path
        )  # Reset if file not exist (or always check)

        # -----------------------------
        # Create subscribers
        # -----------------------------
        self.create_subscription(LaserScan, "/lidar/scan", self.lidar_callback, 20)
        self.create_subscription(Float32, "/lidar/tilt_angle", self.tilt_callback, 20)
        self.create_subscription(Imu, "/imu/data", self.imu_callback, 20)
        self.create_subscription(Odometry, "/odom", self.odom_callback, 20)

        # Ensure directory exists
        self._ensure_csv_directory()

        self.get_logger().info(
            "FusionNode (Python) started – logging sensor data for WebApp 3D Mapping"
        )

    # -----------------------------
    # Ensure directory exists
    # -----------------------------
    def _ensure_csv_directory(self):
        dir_path = os.path.dirname(self.csv_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

    # -----------------------------
    # Callbacks
    # -----------------------------
    def lidar_callback(self, msg: LaserScan):
        self.latest_lidar = msg
        self.try_write_csv()

    def tilt_callback(self, msg: Float32):
        self.latest_phi_deg = float(msg.data)
        self.try_write_csv()

    def imu_callback(self, msg: Imu):
        """Extract yaw (psi_r) in radians from quaternion"""
        q = msg.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.latest_imu = math.atan2(siny_cosp, cosy_cosp)  # radians
        self.try_write_csv()

    def odom_callback(self, msg: Odometry):
        self.latest_odom = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.try_write_csv()

    # -----------------------------
    # Normalize yaw to [-180, 180] degrees
    # -----------------------------
    def _normalize_yaw(self, yaw_deg: float) -> float:
        """Normalize yaw to [-180, 180] for easier reading in WebApp"""
        yaw_deg = yaw_deg % 360
        if yaw_deg > 180:
            yaw_deg -= 360
        elif yaw_deg < -180:
            yaw_deg += 360
        return yaw_deg

    # -----------------------------
    # Write CSV when ALL data available
    # -----------------------------
    def try_write_csv(self):
        # Wait for all sensors
        if not all(
            [
                self.latest_lidar,
                self.latest_phi_deg is not None,
                self.latest_imu is not None,
                self.latest_odom,
            ]
        ):
            # Thêm phần log chờ ở đây
            missing = []
            if not self.latest_lidar:
                missing.append("LIDAR")
            if self.latest_phi_deg is None:
                missing.append("TILT")
            if not self.latest_imu:
                missing.append("IMU")
            if not self.latest_odom:
                missing.append("ODOM")
            if missing:
                self.get_logger().warn(f"Waiting for: {', '.join(missing)}")
            return  # Thiếu → KHÔNG GHI

        # Phần còn lại giữ nguyên...
        lidar: LaserScan = self.latest_lidar
        x_r, y_r = self.latest_odom
        psi_r_rad = self.latest_imu
        psi_r_deg = self._normalize_yaw(
            math.degrees(psi_r_rad)
        )  # Convert to normalized degrees

        # ISO 8601 UTC timestamp with milliseconds and 'Z'
        timestamp_str = (
            datetime.now(timezone.utc).isoformat(timespec="milliseconds") + "Z"
        )

        # Build lidar data: theta0_deg, r0_m, theta1_deg, r1_m, ...
        arr = []
        angle = lidar.angle_min
        for r in lidar.ranges:
            theta_deg = math.degrees(angle)
            arr.append(theta_deg)
            arr.append(r)
            angle += lidar.angle_increment

        # Final row
        row = [timestamp_str, self.latest_phi_deg] + arr + [x_r, y_r, psi_r_deg]

        # Open file in append mode
        write_header = not self.header_written

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)

            # Write header only once (first valid frame or if file new)
            if write_header:
                num_beams = len(lidar.ranges)
                # Interleaved header: theta0_deg, r0_m, theta1_deg, r1_m, ...
                header = ["timestamp", "phi_deg"]
                for i in range(num_beams):
                    header.append(f"theta{i}_deg")
                    header.append(f"r{i}_m")
                header += ["x_r", "y_r", "psi_r"]
                writer.writerow(header)
                self.header_written = True
                self.get_logger().info(
                    f"CSV initialized: {num_beams} lidar beams → {self.csv_path}"
                )

            # Write data row
            writer.writerow(row)

        self.get_logger().info(
            f"[LOG] wrote frame: {len(lidar.ranges)} points, "
            f"phi={self.latest_phi_deg:.1f}°, psi={psi_r_deg:.1f}°"
        )


def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("FusionNode stopped by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
