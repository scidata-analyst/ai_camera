from __future__ import annotations

import time

from ..attributes.ports import FaceAttributeRecognizer
from ..geometry import sharpness
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
    def __init__(self, detector: FaceDetector, max_faces: int,
                 detect_every: int, recognizer: FaceAttributeRecognizer,
                 headpose: HeadPoseAnalyzer | None):
        self.detector = detector
        self.max_faces = int(max_faces)
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

    def detect(self, frame, source_frame) -> list[Face]:
        self._face_no += 1
        run = (self._face_no % self.detect_every == self.detect_every - 1) \
            or not self._last
        if run:
            started = time.perf_counter()
            faces = self.detector.detect(source_frame)
            self.last_ms = round((time.perf_counter() - started) * 1000, 1)
            if self.max_faces > 0 and len(faces) > self.max_faces:
                faces = sorted(
                    faces,
                    key=lambda face: -((face["box"][2] - face["box"][0]) *
                                       (face["box"][3] - face["box"][1])),
                )[:self.max_faces]
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
        self.recognizer.fill(faces)
        return faces
