from __future__ import annotations

import time

import numpy as np

from ...config.constants import AGE_INPUT_SIZE, FAIRFACE_AGE_BINS, FAIRFACE_MEAN, FAIRFACE_STD
from ...config.settings import Settings
from ...domain.attributes.ports import AttributeEstimator
from ...domain.geometry import crop_face
from ...domain.models import AttributeSample, Face
from .runtime import rknn_infer, rknn_load


class RknnFairFaceAgeEstimator(AttributeEstimator):
    """FairFace RKNN. Age only; gender comes from InsightFace."""

    def __init__(self, settings: Settings):
        self.rknn_path = settings.age_rknn_path
        self.rknn      = None
        self.runs      = 0
        self.last_ms   = 0.0
        self._mean     = np.asarray(FAIRFACE_MEAN, dtype=np.float32)
        self._std      = np.asarray(FAIRFACE_STD, dtype=np.float32)

        if settings.age_backend == "off":
            return
        if settings.age_backend not in ("rknn", ""):
            raise RuntimeError(f"AGE_GENDER_BACKEND={settings.age_backend!r} is disabled in NPU-only mode")

        self.rknn, core = rknn_load(self.rknn_path, "age")
        print(f"[age] FairFace RKNN NPU active: {self.rknn_path} core={core}")

    def describe(self) -> dict:
        return {
            "backend": "rknn" if self.rknn is not None else "off",
            "rknn_model": self.rknn_path,
            "input_size": AGE_INPUT_SIZE,
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

        aligned = crop_face(frame, face, mode, AGE_INPUT_SIZE)
        rgb     = aligned[:, :, ::-1].astype(np.float32) / 255.0
        rgb     = (rgb - self._mean) / self._std
        blob    = np.ascontiguousarray(rgb.transpose(2, 0, 1)[np.newaxis, ...], dtype=np.float32)
        started = time.perf_counter()

        try:
            outputs = rknn_infer(self.rknn, blob, "nchw", "float32")

            if not outputs:
                raise RuntimeError("age RKNN returned no outputs")

            arrays = [np.asarray(value, dtype=np.float32).reshape(-1) for value in outputs]
            bins   = next((array for array in arrays if array.size == FAIRFACE_AGE_BINS.size), None)
            age    = None if bins is None else float(np.dot(bins, FAIRFACE_AGE_BINS))

            if age is None:
                scalar = next((array for array in arrays if array.size == 1), None)

                if scalar is None:
                    raise RuntimeError(f"expected FairFace age output, got {[array.shape for array in arrays]}")
                age = float(scalar[0])

            self.runs += 1

            return AttributeSample(
                raw_age=age,
                gender_logits=(0.0, 0.0),
                align=mode,
                sharpness=float(face.get("sharpness", 0.0)),
            )
        except Exception as exc:
            print(f"[age] inference failed: {exc}")

            return None
        finally:
            self.last_ms = round((time.perf_counter() - started) * 1000, 1)
