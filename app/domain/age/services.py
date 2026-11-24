from __future__ import annotations

import numpy as np

from ..models import AttributeSample, Face


class AgeService:
    def __init__(self, scale: float = 1.0, bias: float = 0.0):
        self.scale = float(scale)
        self.bias = float(bias)

    def apply(self, face: Face, sample: AttributeSample) -> None:
        raw_age = float(sample.raw_age)
        age = np.clip(np.rint(raw_age * self.scale + self.bias), 0, 120)
        face.update({
            "age": int(age),
            "raw_age": round(raw_age, 1),
            "align": sample.align,
        })

    def describe(self) -> dict:
        return {"scale": self.scale, "bias": self.bias}
