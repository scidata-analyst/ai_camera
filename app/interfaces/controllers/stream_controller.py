from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse, StreamingResponse

from ...application.frame_processor import FrameProcessor
from ...application.ports import FrameSource
from ...config.settings import Settings
from ..auth import require_token
from ..flip_state import FlipState

INDEX_PATH = Path(__file__).with_name("index.html")
FONT_PATH  = Path(__file__).with_name("UbuntuSans.ttf")

public_router = APIRouter()
router        = APIRouter(dependencies=[Depends(require_token)])


@public_router.get("/ubuntu-sans.ttf")
def ubuntu_sans():
    return FileResponse(FONT_PATH, media_type="font/ttf")


@router.get("/")
def index():
    return FileResponse(INDEX_PATH, media_type="text/html")


@router.get("/video_feed")
def video_feed(request: Request):
    return StreamingResponse(
        request.app.state.runtime.stream.frames(),
        media_type="multipart/x-mixed-replace; boundary=frame")


class StreamController:
    def __init__(self, camera: FrameSource, processor: FrameProcessor, settings: Settings, flip: FlipState):
        self.camera    = camera
        self.processor = processor
        self.settings  = settings
        self.flip      = flip
        self._lock     = threading.Lock()
        self._latest   = None

    def start(self) -> None:
        threading.Thread(target=self._infer_loop, daemon=True).start()

    def _infer_loop(self) -> None:
        previous = time.time()

        while True:
            try:
                frame = self.flip.apply(self.camera.read())
                annotated, _ = self.processor.infer(frame)
                now      = time.time()
                instant  = 1.0 / max(now - previous, 1e-6)
                previous = now

                self.processor.infer_fps = self.processor.infer_fps * 0.9 + instant * 0.1

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
