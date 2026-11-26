from __future__ import annotations

import time

import numpy as np

from ...config.constants import GENDER_INPUT_SIZE
from ...config.settings import Settings
from ...domain.attributes.ports import AttributeEstimator
from ...domain.geometry import crop_face
from ...domain.models import AttributeSample, Face
from .runtime import rknn_infer, rknn_load


class RknnGenderEstimator(AttributeEstimator):
    """InsightFace buffalo_l genderage RKNN. Gender only; age comes from FairFace."""

    def __init__(self, settings: Settings):
        self.rknn_path = settings.gender_rknn_path
        self.rknn      = None
        self.runs      = 0
        self.last_ms   = 0.0

        if settings.age_backend == "off":
            return
        if settings.age_backend not in ("rknn", ""):
            raise RuntimeError(f"AGE_GENDER_BACKEND={settings.age_backend!r} is disabled in NPU-only mode")

        self.rknn, core = rknn_load(self.rknn_path, "gender")
        print(f"[gender] InsightFace RKNN NPU active: {self.rknn_path} core={core}")

    def describe(self) -> dict:
        return {
            "backend": "rknn" if self.rknn is not None else "off",
            "rknn_model": self.rknn_path,
            "input_size": GENDER_INPUT_SIZE,
            "loaded": self.rknn is not None,
            "runs": self.runs,
            "last_ms": self.last_ms,
        }

    def estimate(self, frame: np.ndarray, face: Face, mode: str) -> AttributeSample | None:
        if self.rknn is None:
            return None

        x1, y1, x2, y2 = map(float, face["box"])

        if x2 - x1 < 1 or y2 - y1 < 1:
            return None

        aligned = crop_face(frame, face, "insightface", GENDER_INPUT_SIZE)
        rgb     = aligned[:, :, ::-1].astype(np.float32)
        blob    = np.ascontiguousarray(rgb[np.newaxis, ...], dtype=np.float32)
        started = time.perf_counter()

        try:
            outputs = rknn_infer(self.rknn, blob, "nhwc", "float32")

            if not outputs:
                raise RuntimeError("gender RKNN returned no outputs")

            values = np.asarray(outputs[0], dtype=np.float32).reshape(-1)

            if values.size < 2:
                raise RuntimeError(f"expected 2 gender outputs, got {values.shape}")

            self.runs += 1

            # InsightFace: gender = argmax(out[0:2]); 1 = male, 0 = female.
            return AttributeSample(
                raw_age=0.0,
                gender_logits=(float(values[0]), float(values[1])),
                align="insightface",
                sharpness=float(face.get("sharpness", 0.0)),
            )
        except Exception as exc:
            print(f"[gender] inference failed: {exc}")

            return None
        finally:
            self.last_ms = round((time.perf_counter() - started) * 1000, 1)
