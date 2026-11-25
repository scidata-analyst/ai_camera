from __future__ import annotations

import time

from ..geometry import iou
from ..models import Detection, Face
from .ports import PersonDetector


class HumanDetectionService:
    def __init__(self, detector: PersonDetector, conf: float, min_height: int, require_face: bool, detect_every: int, stale_sec: float):
        self.detector             = detector
        self.conf                 = float(conf)
        self.min_height           = int(min_height)
        self.require_face_enabled = bool(require_face)
        self.detect_every         = max(1, int(detect_every))
        self.stale_sec            = float(stale_sec)
        self._frame_no            = 0
        self._last: list[Detection] = []
        self._last_time  = 0.0
        self._has_result = False
        self._tracks: list = []
        self._next_id = 0
        self.last_ms  = 0.0

    @property
    def loaded(self) -> bool:
        return bool(self.detector.describe().get("loaded"))

    def describe(self) -> dict:
        return self.detector.describe()

    def _keep(self, detection: Detection) -> bool:

        if detection["kind"] != "human":
            return False
        x1, y1, x2, y2 = detection["box"]
        return (detection["conf"] >= self.conf and (y2 - y1) >= self.min_height)

    def _assign_ids(self, detections: list[Detection]) -> None:
        used = set()

        for detection in detections:
            box = detection["box"]
            best, best_iou = None, 0.0

            for track in self._tracks:

                if track["id"] in used:
                    continue
                overlap = iou(box, track["box"])

                if overlap > best_iou:
                    best, best_iou = track, overlap

            if best is not None and best_iou >= 0.3:
                detection["person_id"] = best["id"]
                best["box"]            = box
                used.add(best["id"])
            else:
                self._next_id += 1
                detection["person_id"] = self._next_id
                self._tracks.append({"id": self._next_id, "box": box})
                used.add(self._next_id)
        self._tracks = [track for track in self._tracks if track["id"] in used]

    def detect(self, frame) -> list[Detection]:
        self._frame_no += 1
        run = (self._frame_no % self.detect_every == 0) or not self._has_result

        if not run and self._has_result and time.time() - self._last_time > self.stale_sec:
            self._last = []
            run        = True

        if run:
            started      = time.perf_counter()
            results      = self.detector.detect(frame)
            self.last_ms = round((time.perf_counter() - started) * 1000, 1)
            self._last   = [dict(d) for d in results if self._keep(d)]
            self._assign_ids(self._last)
            self._last_time  = time.time()
            self._has_result = True
        else:
            self._last = [dict(d) for d in self._last if self._keep(d)]
        return [dict(d) for d in self._last]

    def require_face(self, detections: list[Detection], faces: list[Face]) -> list[Detection]:

        if not self.require_face_enabled:
            return detections

        def has_face(detection: Detection) -> bool:
            x1, y1, x2, y2 = detection["box"]
            return any(x1 <= face["box"][0] <= x2 and y1 <= face["box"][1] <= y2 for face in faces)

        return[d for d in detections if d["kind"] != "human" or has_face(d)]
