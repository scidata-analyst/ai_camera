from __future__ import annotations

from ..domain.age.services import AgeService
from ..domain.attributes.ports import AttributeEstimator, FaceAttributeRecognizer
from ..domain.aggregation.services import AttributeAggregator
from ..domain.gender.services import GenderService
from ..domain.models import Face
from ..domain.face.services import FaceQualityService


class AttributeRecognitionService(FaceAttributeRecognizer):
    def __init__(self, estimator: AttributeEstimator,
                 quality: FaceQualityService, age: AgeService,
                 gender: GenderService, aggregator: AttributeAggregator,
                 min_size: int, align_mode: str):
        self.estimator = estimator
        self.quality = quality
        self.age = age
        self.gender = gender
        self.aggregator = aggregator
        self.min_size = int(min_size)
        self.align_mode = align_mode

    def recognize(self, frame, faces: list[Face]) -> None:
        for face in faces:
            x1, y1, x2, y2 = face["box"]
            if min(x2 - x1, y2 - y1) < self.min_size:

                face.update({"gender": "Unknown", "gconf": 0.0, "age": 0})
                continue
            self.quality.evaluate(frame, face)
            if face.get("blurry"):

                continue
            sample = self.estimator.estimate(frame, face, self.align_mode)
            if sample is None:

                face.update({"gender": "Unknown", "gconf": 0.0, "age": 0})
                continue
            self.age.apply(face, sample)
            self.gender.apply(face, sample)
        self.aggregator.add(faces)

    def fill(self, faces: list[Face]) -> list[Face]:
        return self.aggregator.fill(faces)

    def results(self) -> dict:
        return self.aggregator.results()

    def describe(self) -> dict:
        details = dict(self.estimator.describe())
        details.update({
            "align": self.align_mode,
            "blur": self.quality.blur_threshold,
        })
        details.update(self.age.describe())
        return details
