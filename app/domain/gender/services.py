from __future__ import annotations

import numpy as np

from ..models import AttributeSample, Face


class GenderService:
    @staticmethod
    def apply(face: Face, sample: AttributeSample) -> None:
        logits = np.asarray(sample.gender_logits, dtype=np.float32)
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs /= max(float(np.sum(probs)), 1e-12)
        gender_id = int(np.argmax(probs))
        face.update({
            "gender": "Male" if gender_id == 1 else "Female",
            "gconf": round(float(probs[gender_id]), 3),
        })
