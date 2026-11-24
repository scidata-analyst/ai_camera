from __future__ import annotations

import threading
import time

import cv2
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from ...application.frame_processor import FrameProcessor
from ...application.ports import FrameSource
from ...config.settings import Settings
from ..auth import require_token
from ..flip_state import FlipState

HTML_PAGE = """
<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Ubuntu+Sans:wght@400;500;700&display=swap" rel="stylesheet">
<title>AI Camera - Human + Face</title>
<style>
html,body{height:100%}
body{background:#000;color:#eee;font-family:'Ubuntu Sans','Ubuntu',sans-serif;text-align:center;margin:0;
     display:flex;flex-direction:column;height:100vh;overflow:hidden}
header{padding:8px;background:#000;font-size:18px;flex:0 0 auto}
img#vid{flex:1 1 auto;min-height:0;width:100vw;object-fit:contain;background:#000}
.bar{flex:0 0 auto;padding:4px;font-size:13px}
.panel{background:#1e1e1e;display:inline-block;padding:6px 14px;border-radius:8px}
.panel b{color:#4fc3f7}
.info{color:#aaa}
a{color:#4fc3f7}
button{margin:0 4px}
</style></head>
<body>
<header>AI Camera - Human / Face / Age / Gender (YOLOv11n + RetinaFace)</header>
<img id="vid" src="/video_feed" />
<div class="bar"><span class="panel" id="s">loading stats... (1s update)</span></div>
<div class="bar">
<button onclick="setFlip('h')">Flip H: <span id="fh">-</span></button>
<button onclick="setFlip('v')">Flip V: <span id="fv">-</span></button>
<button onclick="resetFlip()">Reset</button>
<span class="info">Green Human | Cyan Face | <b>Green [LOOK] = Looking</b> |
<a href="/api/stats">stats</a> | <a href="/api/faces">faces</a> | <a href="/api/health">health</a></span>
</div>
<script>
const TOKEN = new URLSearchParams(location.search).get('token') || '';
const HEADERS = TOKEN ? {'Authorization': 'Bearer ' + TOKEN} : {};
document.getElementById('vid').src = '/video_feed' +
  (TOKEN ? '?token=' + encodeURIComponent(TOKEN) : '');
async function tick(){
  try{
    const r = await fetch('/api/stats', {headers: HEADERS}); const j = await r.json();
    const s = j.smoothed;
    document.getElementById('s').innerHTML =
      `Stable(1s) <b>H:${s.stable_human} F:${s.stable_face}</b> ` +
      `M:${s.stable_male} W:${s.stable_female} Look:${j.instant.looking} | ` +
      `infer:${j.infer_fps}fps age/gender:${j.age_gender.runs} ` +
      `${j.age_gender.last_ms}ms`;
    document.getElementById('fh').innerText = j.flip.h ? 'ON' : 'OFF';
    document.getElementById('fv').innerText = j.flip.v ? 'ON' : 'OFF';
  }catch(e){}
}
async function setFlip(axis){
  await fetch('/api/flip/toggle?axis='+axis, {method:'POST', headers: HEADERS});
  tick();
}
async function resetFlip(){
  await fetch('/api/flip/set?h=0&v=0', {method:'POST', headers: HEADERS});
  tick();
}
setInterval(tick, 1000); tick();
</script>
</body></html>
"""

router = APIRouter(dependencies=[Depends(require_token)])


@router.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


@router.get("/video_feed")
def video_feed(request: Request):
    return StreamingResponse(
        request.app.state.runtime.stream.frames(),
        media_type="multipart/x-mixed-replace; boundary=frame")


class StreamController:
    def __init__(self, camera: FrameSource, processor: FrameProcessor,
                 settings: Settings, flip: FlipState):
        self.camera = camera
        self.processor = processor
        self.settings = settings
        self.flip = flip
        self._lock = threading.Lock()
        self._latest = None

    def start(self) -> None:
        threading.Thread(target=self._infer_loop, daemon=True).start()

    def _infer_loop(self) -> None:
        previous = time.time()
        while True:
            try:
                frame = self.flip.apply(self.camera.read())
                annotated, _ = self.processor.infer(frame)
                now = time.time()
                instant = 1.0 / max(now - previous, 1e-6)
                previous = now
                self.processor.infer_fps = (
                    self.processor.infer_fps * 0.9 + instant * 0.1)
                with self._lock:
                    self._latest = annotated
                time.sleep(0.005)
            except Exception as exc:
                print(f"[infer] {exc}")
                time.sleep(0.2)

    def frames(self):
        interval = 1.0 / max(1.0, self.settings.stream_fps)
        while True:
            started = time.time()
            with self._lock:
                reference = self._latest
            image = reference.copy() if reference is not None else None
            if image is None:
                image = self.flip.apply(self.camera.read())
            ok, buffer = cv2.imencode(
                ".jpg", image,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.settings.jpeg_quality])
            if not ok:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buffer.tobytes() + b"\r\n")
            elapsed = time.time() - started
            time.sleep(max(0.0, interval - elapsed))
