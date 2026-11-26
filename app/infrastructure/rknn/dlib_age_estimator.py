from __future__ import annotations

import time

import cv2
import numpy as np

from ...config.constants import (
    AGE_INPUT_SIZE,
    AGE_ONLY_INPUT_SIZE,
    DLIB_AGE_MEAN,
    DLIB_AGE_SCALE,
    DLIB_AGE_WEIGHTS,
)
from ...config.settings import Settings
from ...domain.geometry import crop_face
from ...domain.models import Face
from .runtime import rknn_infer, rknn_load


class RknnDlibAgeEstimator:
    def __init__(self, settings: Settings):
        self.rknn_path = settings.age_only_rknn_path
        self.rknn = None

        if settings.age_backend == "rknn":
            self.rknn, core = rknn_load(self.rknn_path, "age-dlib")
            print(f"[age-dlib] RKNN NPU active: {self.rknn_path} core={core}")
        self.runs    = 0
        self.last_ms = 0.0
        self._mean   = np.asarray(DLIB_AGE_MEAN, dtype=np.float32)

    def describe(self) -> dict:
        return {
            "backend": "rknn" if self.rknn is not None else "off",
            "rknn_model": self.rknn_path,
            "input_size": AGE_ONLY_INPUT_SIZE,
            "loaded": self.rknn is not None,
            "runs": self.runs,
            "last_ms": self.last_ms,
        }

    def estimate_age(self, frame: np.ndarray, face: Face, mode: str) -> float | None:
        if self.rknn is None:
            return None
        x1, y1, x2, y2 = map(float, face["box"])

        if x2 - x1 < 1 or y2 - y1 < 1:
            return None
        aligned = crop_face(frame, face, mode, AGE_INPUT_SIZE)
        chip    = cv2.resize(aligned, (AGE_ONLY_INPUT_SIZE, AGE_ONLY_INPUT_SIZE), interpolation=cv2.INTER_AREA)
        rgb     = chip[:, :, ::-1].astype(np.float32)
        rgb     = (rgb - self._mean) / DLIB_AGE_SCALE
        blob    = np.ascontiguousarray(rgb.transpose(2, 0, 1)[np.newaxis, ...], dtype=np.float32)
        started = time.perf_counter()

        try:
            outputs = rknn_infer(self.rknn, blob, "nchw", "float32")

            if not outputs:
                raise RuntimeError("age-dlib RKNN returned no outputs")

            for output in outputs:
                array = np.asarray(output, dtype=np.float32).reshape(-1)

                if array.size == DLIB_AGE_WEIGHTS.size:
                    self.runs += 1

                    return float(np.dot(array, DLIB_AGE_WEIGHTS))
            raise RuntimeError(
                f"expected {DLIB_AGE_WEIGHTS.size} age classes, got {[np.asarray(value).shape for value in outputs]}"
            )
        except Exception as exc:
            print(f"[age-dlib] inference failed: {exc}")

            return None
        finally:
            self.last_ms = round((time.perf_counter() - started) * 1000, 1)
