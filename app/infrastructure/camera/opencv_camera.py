from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from ...application.ports import FrameSource
from ..rendering.text import draw as draw_text


class OpenCVCamera(FrameSource):
    def __init__(self, source: str, width: int, height: int):
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        self.source                     = source
        self.width                      = int(width)
        self.height                     = int(height)
        self.cap                        = None
        self.frame: np.ndarray | None   = None
        self.running                    = False
        self.lock                       = threading.Lock()
        self._connected                 = False
        self._connect()

    @property
    def connected(self) -> bool:

        return self._connected

    def _connect(self) -> None:
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass

        self.cap = cv2.VideoCapture(self.source)

        if self.cap.isOpened():
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            try:
                self.cap.set(cv2.CAP_PROP_FOURCC,
                             cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                pass

            try:
                self.cap.set(cv2.CAP_PROP_FPS, 30)
            except Exception:
                pass

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._connected = True
        else:
            self._connected = False

    def start(self) -> None:
        if self.running:

            return
        
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _synthetic(self) -> np.ndarray:
        image       = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        position    = int(time.time() * 2) % self.width

        cv2.rectangle(image, (position - 80, 100), (position, 300), (255, 255, 255), -1)
        draw_text(image, (20, 20), "NO CAMERA - synthetic feed", 20, (0, 255, 255), 1)
        draw_text(image, (20, 48), f"src={self.source} set CAMERA_SOURCE", 16, (200, 200, 200))

        return image

    def _loop(self) -> None:
        while self.running:
            if self._connected and self.cap is not None:
                ok, frame = self.cap.read()

                if ok and frame is not None:

                    with self.lock:
                        self.frame = frame
                    time.sleep(0.005)
                else:
                    self._connected = False
                    time.sleep(0.5)
                    self._connect()
            else:

                with self.lock:
                    self.frame = self._synthetic()
                time.sleep(0.05)

    def read(self) -> np.ndarray:

        with self.lock:
            if self.frame is not None:

                return self.frame.copy()
            
        return self._synthetic()

    def stop(self) -> None:
        self.running = False
        
        if self.cap is not None:
            self.cap.release()
