from __future__ import annotations

import time

from ..attributes.ports import FaceAttributeRecognizer
from ..geometry import iou, sharpness
from ..headpose.ports import HeadPoseAnalyzer
from ..models import Face
from .ports import FaceDetector


class FaceQualityService:
    def __init__(self, blur_threshold: float):
        self.blur_threshold = float(blur_threshold)

    def evaluate(self, frame, face: Face) -> None:
        value = sharpness(frame, face["box"])
        face["sharpness"] = round(value, 1)
        if self.blur_threshold > 0 and value < self.blur_threshold:
            face.update({
                "gender": "Unknown",
                "gconf": 0.0,
                "age": 0,
                "blurry": True,
            })
            return
        face["blurry"] = False


class FaceDetectionService:
    def __init__(self, detector: FaceDetector, detect_every: int,
                 recognizer: FaceAttributeRecognizer,
                 headpose: HeadPoseAnalyzer | None):
        self.detector = detector
        self.detect_every = max(1, int(detect_every))
        self.recognizer = recognizer
        self.headpose = headpose
        self._face_no = 0
        self._last: list[Face] = []
        self.last_ms = 0.0
        self.age_gender_ms = 0.0

    @property
    def loaded(self) -> bool:
        return bool(self.detector.describe().get("loaded"))

    def describe(self) -> dict:
        return self.detector.describe()

    def _detect_in_humans(self, source_frame, humans) -> list[Face]:
        height, width = source_frame.shape[:2]
        found: list[Face] = []
        for human in humans:
            hx1, hy1, hx2, hy2 = (int(value) for value in human["box"])
            x1, y1 = max(0, hx1), max(0, hy1)
            x2, y2 = min(width, hx2), min(height, hy2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            for face in self.detector.detect(source_frame[y1:y2, x1:x2]):
                x1f, y1f, x2f, y2f = (int(value) for value in face["box"])
                face["box"] = [x1f + x1, y1f + y1, x2f + x1, y2f + y1]
                landmarks = face.get("landmarks")
                if landmarks:
                    face["landmarks"] = [
                        [int(px) + x1, int(py) + y1] for px, py in landmarks]
                face["human_box"] = list(human["box"])
                person_id = human.get("person_id")
                if person_id is not None:
                    face["person_id"] = person_id
                found.append(face)
        unique: list[Face] = []
        for face in found:
            if any(iou(face["box"], other["box"]) > 0.7 for other in unique):
                continue
            unique.append(face)
        return unique

    def detect(self, frame, source_frame, humans=None) -> list[Face]:
        self._face_no += 1
        run = (self._face_no % self.detect_every == self.detect_every - 1) \
            or not self._last
        if run:
            started = time.perf_counter()
            faces = self._detect_in_humans(source_frame, humans or [])
            self.last_ms = round((time.perf_counter() - started) * 1000, 1)
            started = time.perf_counter()
            self.recognizer.recognize(source_frame, faces)
            self.age_gender_ms = round(
                (time.perf_counter() - started) * 1000, 1)
            if self.headpose is not None:
                faces = self.headpose.fill_from_cache(faces)
                self.headpose.submit(source_frame, faces)
            self._last = [dict(face) for face in faces]
        else:
            faces = [dict(face) for face in self._last]
            if self.headpose is not None:
                faces = self.headpose.fill_from_cache(faces)
        self.recognizer.fill(faces)
        return faces
