from __future__ import annotations

import time

from .head_pose import HeadPoseService
from .ports import FrameRenderer
from ..domain.aggregation.services import CountAggregator
from ..domain.attributes.ports import FaceAttributeRecognizer
from ..domain.detection.services import HumanDetectionService
from ..domain.face.services import FaceDetectionService
from ..domain.headpose.ports import HeadPoseAnalyzer


class FrameProcessor:
    def __init__(self, human_detection: HumanDetectionService,
                 face_detection: FaceDetectionService | None,
                 attributes: FaceAttributeRecognizer | None,
                 headpose: HeadPoseAnalyzer | None,
                 count_aggregator: CountAggregator,
                 renderer: FrameRenderer):
        self.human_detection = human_detection
        self.face_detection = face_detection
        self.attributes = attributes
        self.headpose = headpose
        self.count_aggregator = count_aggregator
        self.renderer = renderer
        self.last_counts = {"human": 0, "non_human": 0, "face": 0,
                            "male": 0, "female": 0, "fps": 0.0}
        self.last_faces = []
        self.smoothed = count_aggregator.current
        self.infer_fps = 0.0
        self._t_prev = time.time()
        self._fps_smooth = 0.0

    def infer(self, frame):
        source_frame = frame.copy()
        started = time.perf_counter()

        detections = self.human_detection.detect(frame)
        faces = []
        if self.face_detection is not None:
            faces = self.face_detection.detect(frame, source_frame)
            self.renderer.draw_faces(frame, faces)
        detections = self.human_detection.require_face(detections, faces)
        self.renderer.draw_humans(frame, detections, faces)

        human_n = sum(1 for detection in detections
                      if detection["kind"] == "human")
        non_human_n = 0
        face_n = len(faces)
        male_n = sum(1 for face in faces if face.get("gender") == "Male")
        female_n = sum(1 for face in faces if face.get("gender") == "Female")
        look_n = sum(1 for face in faces if face.get("looking"))
        self.last_faces = faces

        now = time.time()
        instant_fps = 1.0 / max(now - self._t_prev, 1e-6)
        self._t_prev = now
        self._fps_smooth = self._fps_smooth * 0.9 + instant_fps * 0.1

        self.smoothed = self.count_aggregator.add(
            human_n, non_human_n, face_n, male_n, female_n)
        stable_human = self.smoothed["stable_human"]
        stable_non_human = self.smoothed["stable_non_human"]
        stable_face = self.smoothed.get("stable_face", face_n)
        stable_male = self.smoothed.get("stable_male", male_n)
        stable_female = self.smoothed.get("stable_female", female_n)
        human_stats = self.smoothed["human"]
        self.renderer.draw_overlay(frame, {
            "human_n": human_n,
            "face_n": face_n,
            "male_n": male_n,
            "female_n": female_n,
            "non_human_n": non_human_n,
            "look_n": look_n,
            "stable_human": stable_human,
            "stable_non_human": stable_non_human,
            "stable_face": stable_face,
            "stable_male": stable_male,
            "stable_female": stable_female,
            "mean": human_stats["mean"],
            "median": human_stats["median"],
            "mode": human_stats["mode"],
            "fps": self._fps_smooth,
        })

        timing = {
            "yolo_ms": self.human_detection.last_ms,
            "face_ms": (self.face_detection.last_ms
                        if self.face_detection is not None else 0.0),
            "age_gender_ms": (self.face_detection.age_gender_ms
                              if self.face_detection is not None else 0.0),
            "infer_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        self.last_counts = {
            "human": human_n,
            "non_human": non_human_n,
            "face": face_n,
            "male": male_n,
            "female": female_n,
            "looking": look_n,
            "fps": round(self._fps_smooth, 1),
            "t_yolo_ms": timing["yolo_ms"],
            "t_face_ms": timing["face_ms"],
            "t_age_gender_ms": timing["age_gender_ms"],
            "t_infer_ms": timing["infer_ms"],
        }
        if isinstance(self.headpose, HeadPoseService):
            self.last_counts["t_bg_ms"] = self.headpose.last_ms
            self.last_counts["bg_pending"] = self.headpose.pending
        return frame, detections
