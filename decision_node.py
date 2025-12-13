# robot/decision_node.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decision Node
- Subscribe /detections (vision_msgs/Detection2DArray)
- Simple rule-based:
    - Nếu person detected with confidence > threshold -> send alert
    - Use optional pose info to include robot heading
- Publish:
    - /alert (std_msgs/String)   JSON string: {"type":"person","count":1,"boxes":[...],"confidence":[...],"timestamp":...}
    - /alert_level (std_msgs/Int8)  e.g. 0=no alert,1=info,2=warning
"""
# Import các thư viện cần thiết
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int8
from vision_msgs.msg import Detection2DArray
import json, time


class DecisionNode(Node):
    # Hàm khởi tạo của DecisionNode
    def __init__(self):
        super().__init__("decision_node")
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("front_fov_px", 640)  # Hiện tại không sử dụng
        self.alert_pub = self.create_publisher(String, "/alert", 10)
        self.alert_level_pub = self.create_publisher(Int8, "/alert_level", 10)
        self.create_subscription(
            Detection2DArray, "/detections", self.detections_cb, 10
        )
        self.get_logger().info("DecisionNode started.")

    # Callback xử lý dữ liệu detections
    def detections_cb(self, msg: Detection2DArray):
        """
        Callback xử lý kết quả detection từ perception node.

        - Lọc các detection có class 'person' và confidence >= threshold
        - Nếu không phát hiện người: publish alert_level = 0
        - Nếu phát hiện người: publish alert (JSON) và alert_level = warning
        """
        try:
            conf_th = float(self.get_parameter("confidence_threshold").value)
            persons = []
            for det in msg.detections:
                # Kiểm tra nếu có kết quả detection
                if det.results and len(det.results) > 0:
                    hyp = det.results[0].hypothesis
                    score = float(hyp.score) if hasattr(hyp, "score") else 0.0
                    cid = hyp.id if hasattr(hyp, "id") else ""
                    if (
                        cid == "person" or str(cid).lower() == "person"
                    ) and score >= conf_th:
                        # Lấy thông tin bounding box
                        cx = det.bbox.center.position.x
                        cy = det.bbox.center.position.y
                        w = det.bbox.size_x
                        h = det.bbox.size_y
                        persons.append(
                            {
                                "cx": float(cx),
                                "cy": float(cy),
                                "w": float(w),
                                "h": float(h),
                                "score": score,
                            }
                        )
            if len(persons) == 0:
                # Không có người được phát hiện
                level = Int8()
                level.data = 0
                self.alert_level_pub.publish(level)
                return
            # Có người được phát hiện, publish alert
            payload = {
                "type": "person_detected",
                "count": len(persons),
                "persons": persons,
                "timestamp": time.time(),
            }
            s = String()
            s.data = json.dumps(payload)
            self.alert_pub.publish(s)
            level = Int8()
            level.data = 2  # Cảnh báo
            self.alert_level_pub.publish(level)
            self.get_logger().info(f"Alert published: {len(persons)} person(s)")
        except Exception as e:
            self.get_logger().error(f"decision error: {e}")


# Hàm main để chạy node
def main(args=None):
    rclpy.init(args=args)
    node = DecisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("DecisionNode stopped")
    finally:
        node.destroy_node()
        rclpy.shutdown()
