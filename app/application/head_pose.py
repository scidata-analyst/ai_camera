from __future__ import annotations

import collections
import threading
import time

from ..domain.headpose.ports import HeadPoseAnalyzer
from ..domain.models import Face


class HeadPoseService(HeadPoseAnalyzer):
    def __init__(self, analyzer: HeadPoseAnalyzer | None, enabled: bool):
        self.analyzer = analyzer if enabled else None
        self._lock    = threading.Lock()
        self._pending = collections.deque(maxlen=2)
        self.last_ms  = 0.0

        if self.analyzer is not None:
            threading.Thread(target=self._bg_loop, daemon=True).start()

    @property
    def enabled(self) -> bool:
        return self.analyzer is not None

    @property
    def pending(self) -> bool:
        return bool(self._pending)

    def _bg_loop(self):
        while True:

            try:
                with self._lock:
                    job = self._pending.popleft() if self._pending else None

                if job is None:
                    time.sleep(0.05)
                    continue
                frame, faces = job
                started = time.perf_counter()

                try:
                    faces = self.analyzer.predict_frame(frame, faces)
                    self.analyzer.remember(faces)
                except Exception as exc:
                    print(f"[headpose-bg] {exc}")
                self.last_ms = round((time.perf_counter() - started) * 1000, 1)
            except Exception as exc:
                print(f"[bg] {exc}")
                time.sleep(0.2)

    def submit(self, frame, faces: list[Face]) -> None:

        if self.analyzer is None or not faces:
            return
        stamp   = time.time()
        payload = [{"box": list(face["box"]), "landmarks": face.get("landmarks"), "looking": face.get("looking"), "t": stamp,} for face in faces]
        with self._lock:
            self._pending.append((frame.copy(), payload))

    def predict_frame(self, frame, faces: list[Face]) -> list[Face]:

        if self.analyzer is None:
            return faces
        return self.analyzer.predict_frame(frame, faces)

    def remember(self, faces: list[Face]) -> list[Face]:

        if self.analyzer is None:
            return faces
        return self.analyzer.remember(faces)

    def fill_from_cache(self, faces: list[Face]) -> list[Face]:

        if self.analyzer is None:
            return faces
        return self.analyzer.fill_from_cache(faces)

    def describe(self) -> dict:

        if self.analyzer is None:
            return {"backend": "off", "rknn_model": None, "providers": [], "loaded": False}
        return self.analyzer.describe()
