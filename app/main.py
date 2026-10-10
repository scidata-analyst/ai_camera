"""FastAPI app: localhost URL e live MJPEG + Human/Non-Human label."""
import os
import time

import cv2
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

from .camera import CameraStream
from .detector import HumanDetector

MODEL_NAME = os.getenv("MODEL_NAME", "yolo11n.pt")
CONF = float(os.getenv("CONF_THRESHOLD", "0.5"))
IMGSZ = int(os.getenv("IMG_SIZE", "480"))  # RK3576 CPU er jonno 480 fast
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "2"))  # proti N frame por por infer
JPEG_Q = int(os.getenv("JPEG_QUALITY", "75"))

app = FastAPI(title="AI Camera - Human Detector (RK3576)")

camera = CameraStream()
detector = HumanDetector(model_name=MODEL_NAME, conf=CONF, imgsz=IMGSZ)
camera.start()

_last_annotated = None
_frame_id = 0

HTML_PAGE = """
<!doctype html>
<html><head><meta charset="utf-8"><title>AI Camera - Human Detector</title>
<style>
body{background:#111;color:#eee;font-family:sans-serif;text-align:center;margin:0}
header{padding:12px;background:#000;font-size:20px}
img{max-width:96vw;border:3px solid #333;border-radius:8px;margin-top:10px}
.info{color:#aaa;margin:10px}
a{color:#4fc3f7}
</style></head>
<body>
<header>AI Camera - Human / Non-Human (YOLOv11n) - RK3576</header>
<img src="/video_feed" />
<div class="info">Green box = <b>Human</b> | Orange box = <b>Non-Human</b> |
<a href="/api/stats">stats JSON</a> | <a href="/api/health">health</a></div>
</body></html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


@app.get("/api/health")
def health():
    return {"status": "ok", "camera_connected": camera.connected,
            "model": MODEL_NAME, "conf": CONF, "imgsz": IMGSZ}


@app.get("/api/stats")
def stats():
    return {"camera_connected": camera.connected,
            "detections": detector.last_counts,
            "time": time.time()}


def gen_frames():
    global _last_annotated, _frame_id
    while True:
        frame = camera.read()
        _frame_id += 1
        # RK3576 CPU bachate frame skip
        if _frame_id % FRAME_SKIP == 0 or _last_annotated is None:
            annotated, _ = detector.infer(frame)
            _last_annotated = annotated
        else:
            _last_annotated = frame  # skip frame e raw
        ok, buf = cv2.imencode(".jpg", _last_annotated,
                               [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_Q])
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")
        # ~25 fps cap
        time.sleep(0.02)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(gen_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")
