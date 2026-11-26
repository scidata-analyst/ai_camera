from __future__ import annotations

from ..models import AttributeSample, Face


class AgeService:
    def apply(self, face: Face, sample: AttributeSample) -> None:
        raw_age = float(sample.raw_age)
        face.update({
            "age": int(round(raw_age)),
            "raw_age": round(raw_age, 1),
            "align": sample.align,
        })
