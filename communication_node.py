# robot/communication_node.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Communication Node
- FastAPI + WebSocket server (accessible on Pi5 LAN)
- Subscribes to:
    - /camera/image_web (sensor_msgs/Image) -- encoding=jpeg expected
    - /detections (vision_msgs/Detection2DArray)
    - /alert (std_msgs/String)
- Streams via WebSocket: JSON messages containing base64 jpeg + detection list + alert
- Serves simple HTML client at '/' to view stream
"""

# Import các thư viện cần thiết
import rclpy
from rclpy.node import Node
import threading, asyncio, base64, json, time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String, Int8


# Định nghĩa lớp CommNode kế thừa từ Node của ROS2
class CommNode(Node):

    # Hàm khởi tạo của CommNode
    def __init__(self):

        # Khởi tạo ROS2 node và các biến lưu trữ trạng thái
        super().__init__("communication_node")
        self.declare_parameter("http_port", 8080)
        self._last_image_jpeg = None  # bytes
        self._last_detections = []
        self._last_alert = None
        self._ws_clients = set()

        # Đăng ký các subscriber để nhận dữ liệu từ các topic ROS2
        self.create_subscription(Image, "/camera/image_web", self.image_cb, 10)
        self.create_subscription(Detection2DArray, "/detections", self.detect_cb, 10)
        self.create_subscription(String, "/alert", self.alert_cb, 10)
        self.create_subscription(Int8, "/alert_level", self.alert_level_cb, 10)

        # Thiết lập FastAPI và chạy server uvicorn trong một thread riêng
        self.app = FastAPI()
        self._setup_routes()

        # Lấy cổng HTTP từ tham số và khởi động server trong thread riêng
        port = int(self.get_parameter("http_port").value)
        threading.Thread(target=self._run_uvicorn, args=(port,), daemon=True).start()
        self.get_logger().info(f"CommunicationNode HTTP server starting on port {port}")

    # Hàm callback xử lý dữ liệu hình ảnh từ topic /camera/image_web
    def image_cb(self, msg: Image):
        try:
            # Lưu trữ ảnh dưới dạng JPEG
            if msg.encoding == "jpeg":
                self._last_image_jpeg = bytes(msg.data)
            else:
                # Chuyển đổi ảnh sang JPEG nếu không phải định dạng JPEG
                import cv2
                from cv_bridge import CvBridge

                bridge = CvBridge()
                cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
                ret, jpg = cv2.imencode(
                    ".jpg", cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                )
                if ret:
                    self._last_image_jpeg = jpg.tobytes()
        except Exception as e:
            self.get_logger().warning(f"image_cb error: {e}")

    # Hàm callback xử lý dữ liệu detections từ topic /detections
    def detect_cb(self, msg: Detection2DArray):
        # convert to simple list
        dets = []
        for det in msg.detections:
            info = {
                "cx": det.bbox.center.position.x,
                "cy": det.bbox.center.position.y,
                "w": det.bbox.size_x,
                "h": det.bbox.size_y,
            }
            # try get score
            try:
                if det.results and len(det.results) > 0:
                    info["score"] = float(det.results[0].hypothesis.score)
                    info["id"] = str(det.results[0].hypothesis.id)
            except Exception:
                pass
            dets.append(info)
        self._last_detections = dets

    # Hàm callback xử lý dữ liệu alert từ topic /alert
    def alert_cb(self, msg: String):
        try:
            self._last_alert = json.loads(msg.data)
        except Exception:
            self._last_alert = {"raw": msg.data}

    # Hàm callback xử lý dữ liệu alert_level từ topic /alert_level. Hiện tại không sử dụng
    def alert_level_cb(self, msg):
        # optional
        self._last_alert_level = int(msg.data)

    # Thiết lập các route cho FastAPI
    def _setup_routes(self):
        @self.app.get("/")
        async def index():
            html = """
<!doctype html>
<html>
<head>
  <title>Robot Live</title>
  <style>
    body{background:#111;color:#eee;font-family:sans-serif}
    #img{max-width:100%;}
    #overlay{position:absolute;left:0;top:0}
    #container{position:relative;display:inline-block}
    #alerts{margin-top:8px;padding:8px;background:#222;border-radius:6px}
  </style>
</head>
<body>
<h2>Robot Live Viewer (FastAPI + WebSocket)</h2>
<div id="container">
  <img id="img" src="" />
</div>
<div id="alerts"></div>
<script>
let ws = new WebSocket("ws://"+location.host+"/ws");
let img = document.getElementById("img");
let alerts = document.getElementById("alerts");
ws.onopen = ()=>console.log("WS open");
ws.onmessage = (ev)=>{
  try {
    let msg = JSON.parse(ev.data);
    if(msg.type === "image"){
      img.src = "data:image/jpeg;base64," + msg.image_b64;
    }
    if(msg.type === "alert"){
      alerts.innerText = "ALERT: " + JSON.stringify(msg.payload);
      alerts.style.background = "#550000";
    } else if(msg.type === "info"){
      alerts.innerText = msg.payload;
      alerts.style.background = "#222";
    }
  } catch(e){ console.warn(e) }
};
ws.onclose = ()=>console.log("WS closed");
</script>
</body>
</html>
"""
            return HTMLResponse(html)

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self._ws_clients.add(websocket)
            try:
                while True:
                    # push latest state at 5 Hz or when updated
                    await asyncio.sleep(0.2)
                    payload = {"type": "image"}
                    if self._last_image_jpeg:
                        payload["image_b64"] = base64.b64encode(
                            self._last_image_jpeg
                        ).decode("ascii")
                    payload["detections"] = self._last_detections
                    if self._last_alert:
                        await websocket.send_json(
                            {"type": "alert", "payload": self._last_alert}
                        )
                    await websocket.send_json(payload)
            except WebSocketDisconnect:
                pass
            finally:
                try:
                    self._ws_clients.remove(websocket)
                except Exception:
                    pass

    # Hàm chạy uvicorn server
    def _run_uvicorn(self, port):
        config = uvicorn.Config(
            self.app, host="0.0.0.0", port=port, log_level="warning"
        )
        server = uvicorn.Server(config)
        # run uvicorn (blocking) in this thread
        server.run()


# Hàm main để chạy node
def main(args=None):
    rclpy.init(args=args)
    node = CommNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("CommunicationNode stopped")
    finally:
        node.destroy_node()
        rclpy.shutdown()
