"""FastAPI app: localhost URL e live MJPEG + Human/Face/Gender label."""
import os
import threading
import time

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

from .camera import CameraStream
from .detector import HumanDetector

MODEL_NAME = os.getenv("MODEL_NAME", "yolo11n.pt")
CONF = float(os.getenv("CONF_THRESHOLD", "0.5"))
IMGSZ = int(os.getenv("IMG_SIZE", "384"))  # 320=fast, 480=accurate, 384=balance
JPEG_Q = int(os.getenv("JPEG_QUALITY", "70"))
WINDOW_SEC = float(os.getenv("SMOOTH_WINDOW", "1.0"))  # 1 sec por por stable update
FACE_ENABLE = os.getenv("FACE_ENABLE", "1") == "1"
MIVOLO_ENABLE = os.getenv("MIVOLO_ENABLE", "1") == "1"
FACE_MIN_SIZE = int(os.getenv("FACE_MIN_SIZE", "60"))  # choto face e gender off
FLIP_H = os.getenv("FLIP_H", "0") == "1"
FLIP_V = os.getenv("FLIP_V", "0") == "1"

app = FastAPI(title="AI Camera - Human + Face + Gender (RK3576)")

camera = CameraStream()
detector = HumanDetector(model_name=MODEL_NAME, conf=CONF, imgsz=IMGSZ,
                         window_sec=WINDOW_SEC, face_enable=FACE_ENABLE,
                         mivolo_enable=MIVOLO_ENABLE,
                         face_min_size=FACE_MIN_SIZE)
camera.start()

_flip = {"h": FLIP_H, "v": FLIP_V}


def apply_flip(frame):
    h, v = _flip["h"], _flip["v"]
    if h and v:
        return cv2.flip(frame, -1)
    if h:
        return cv2.flip(frame, 1)
    if v:
        return cv2.flip(frame, 0)
    return frame


# ---- Background inference: stream kokhono infer er jonno block hobe na.
# Infer thread latest frame e detection chaliye result ready rakhe,
# gen_frames sudhu ready image serve kore (~30fps, lag-free).
_out_lock = threading.Lock()
_latest_annotated: np.ndarray | None = None
_infer_fps = 0.0


def _infer_loop():
    global _latest_annotated, _infer_fps
    t_prev = time.time()
    while True:
        try:
            frame = apply_flip(camera.read())
            annotated, _ = detector.infer(frame)
            now = time.time()
            inst = 1.0 / max(now - t_prev, 1e-6)
            t_prev = now
            _infer_fps = _infer_fps * 0.9 + inst * 0.1
            with _out_lock:
                _latest_annotated = annotated
        except Exception as e:
            print(f"[infer] {e}")
            time.sleep(0.2)


threading.Thread(target=_infer_loop, daemon=True).start()

HTML_PAGE = """
<!doctype html>
<html><head><meta charset="utf-8"><title>AI Camera - Human + Face</title>
<style>
body{background:#111;color:#eee;font-family:sans-serif;text-align:center;margin:0}
header{padding:12px;background:#000;font-size:20px}
img{max-width:96vw;border:3px solid #333;border-radius:8px;margin-top:10px}
.info{color:#aaa;margin:10px}
.panel{background:#1e1e1e;display:inline-block;padding:10px 18px;border-radius:8px;margin-top:8px}
.panel b{color:#4fc3f7}
a{color:#4fc3f7}
</style></head>
<body>
<header>AI Camera - Human / Face / Gender (YOLOv11n + YuNet + MiVOLO)</header>
<img src="/video_feed" />
<div><div class="panel" id="s">loading stats... (1s update)</div></div>
<div style="margin:8px">
<button onclick="setFlip('h')">Flip H: <span id="fh">-</span></button>
<button onclick="setFlip('v')">Flip V: <span id="fv">-</span></button>
<button onclick="resetFlip()">Reset</button>
</div>
<div class="info">Green = <b>Human</b> | Orange = <b>Non-Human</b> | Cyan = <b>Face M/F+age</b> |
<a href="/api/stats">stats</a> | <a href="/api/faces">faces</a> | <a href="/api/health">health</a></div>
<script>
async function tick(){
  try{
    const r = await fetch('/api/stats'); const j = await r.json();
    const s = j.smoothed;
    document.getElementById('s').innerHTML =
      `Stable(1s) <b>H:${s.stable_human} F:${s.stable_face}</b> ` +
      `M:${s.stable_male} W:${s.stable_female} | ` +
      `instant M:${j.instant.male} W:${j.instant.female} infer:${j.infer_fps}fps`;
    document.getElementById('fh').innerText = j.flip.h ? 'ON' : 'OFF';
    document.getElementById('fv').innerText = j.flip.v ? 'ON' : 'OFF';
  }catch(e){}
}
async function setFlip(axis){
  await fetch('/api/flip/toggle?axis='+axis, {method:'POST'});
  tick();
}
async function resetFlip(){
  await fetch('/api/flip/set?h=0&v=0', {method:'POST'});
  tick();
}
setInterval(tick, 1000); tick();
</script>
</body></html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


@app.get("/api/health")
def health():
    return {"status": "ok", "camera_connected": camera.connected,
            "model": MODEL_NAME, "conf": CONF, "imgsz": IMGSZ,
            "face_enabled": FACE_ENABLE,
            "face_loaded": detector.face_enabled,
            "face_min_size": FACE_MIN_SIZE,
            "mivolo_enabled": MIVOLO_ENABLE,
            "mivolo_loaded": detector.mivolo_enabled,
            "infer_fps": round(_infer_fps, 1)}


@app.get("/api/faces")
def faces():
    return {"faces": detector.last_faces, "count": detector.last_counts.get("face", 0),
            "time": time.time()}


@app.get("/api/stats")
def stats():
    return {"camera_connected": camera.connected,
            "instant": detector.last_counts,
            "smoothed": detector.smoothed,
            "flip": _flip,
            "infer_fps": round(_infer_fps, 1),
            "window_sec": WINDOW_SEC,
            "time": time.time()}


@app.get("/api/flip")
def get_flip():
    return _flip


@app.post("/api/flip/set")
def set_flip(h: int = None, v: int = None):
    if h is not None:
        _flip["h"] = bool(int(h))
    if v is not None:
        _flip["v"] = bool(int(v))
    return _flip


@app.post("/api/flip/toggle")
def toggle_flip(axis: str = "h"):
    if axis in ("h", "horizontal"):
        _flip["h"] = not _flip["h"]
    elif axis in ("v", "vertical"):
        _flip["v"] = not _flip["v"]
    elif axis in ("both", "180"):
        _flip["h"] = not _flip["h"]
        _flip["v"] = not _flip["v"]
    return _flip


def gen_frames():
    while True:
        with _out_lock:
            img = _latest_annotated
            if img is not None:
                img = img.copy()
        if img is None:
            img = apply_flip(camera.read())  # infer ready howar age raw
        ok, buf = cv2.imencode(".jpg", img,
                               [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_Q])
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")
        time.sleep(0.033)  # ~30fps stream, infer theke independent


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(gen_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")
