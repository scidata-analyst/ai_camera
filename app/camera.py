"""Threaded camera reader with /dev/video0 + fallback."""
import os
import threading
import time

import cv2
import numpy as np


class CameraStream:
    def __init__(self, source=None, width: int = 640, height: int = 480):
        # CAMERA_SOURCE env: "0" / "/dev/video0" / rtsp url / video file
        src = source or os.getenv("CAMERA_SOURCE", "0")
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        self.source = src
        self.width = int(os.getenv("FRAME_WIDTH", str(width)))
        self.height = int(os.getenv("FRAME_HEIGHT", str(height)))
        self.cap = None
        self.frame: np.ndarray | None = None
        self.running = False
        self.lock = threading.Lock()
        self.connected = False
        self._connect()

    def _connect(self):
        self.cap = cv2.VideoCapture(self.source)
        if self.cap.isOpened():
            # LAG FIX: buffer=1 + MJPG -> purono frame jome lag hoy na,
            # sobsomoy latest frame ase. MJPG decode o fast.
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
            self.connected = True
        else:
            self.connected = False

    def start(self):
        if self.running:
            return
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _synthetic(self) -> np.ndarray:
        """Camera na thakle test pattern, jate localhost URL sobsomoy open hoy."""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        t = int(time.time() * 2) % self.width
        cv2.rectangle(img, (t - 80, 100), (t, 300), (255, 255, 255), -1)
        cv2.putText(img, "NO CAMERA - synthetic feed", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(img, f"src={self.source} set CAMERA_SOURCE", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        return img

    def _loop(self):
        while self.running:
            if self.connected and self.cap is not None:
                ok, f = self.cap.read()
                if ok and f is not None:
                    with self.lock:
                        self.frame = f
                    # LAG FIX: camera thread CPU na khak — choto sleep,
                    # infer/stream thread ke GIL charo
                    time.sleep(0.005)
                else:
                    # camera drop -> reconnect once
                    self.connected = False
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

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
