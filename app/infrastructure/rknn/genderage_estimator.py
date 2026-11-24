from __future__ import annotations

import time

import cv2
import numpy as np

from ...config.constants import AGE_INPUT_SIZE
from ...config.settings import Settings
from ...domain.attributes.ports import AttributeEstimator
from ...domain.geometry import crop_face
from ...domain.models import AttributeSample, Face
from .runtime import rknn_infer, rknn_load


class RknnGenderAgeEstimator(AttributeEstimator):
    def __init__(self, settings: Settings):
        if settings.age_backend == "off":
            self.rknn = None
            self.rknn_path = settings.age_rknn_path
        else:
            if settings.age_backend not in ("rknn", ""):
                raise RuntimeError(
                    f"AGE_GENDER_BACKEND={settings.age_backend!r} is disabled "
                    "in NPU-only mode")
            self.rknn_path = settings.age_rknn_path
            self.rknn, core = rknn_load(self.rknn_path, "age-gender")
            print(f"[age-gender] RKNN NPU active: {self.rknn_path} core={core}")
        self.onnx_path = settings.age_onnx_path
        self.runs = 0
        self.last_ms = 0.0

    def describe(self) -> dict:
        return {
            "backend": "rknn" if self.rknn is not None else "off",
            "onnx_model": self.onnx_path,
            "rknn_model": self.rknn_path,
            "loaded": self.rknn is not None,
            "runs": self.runs,
            "last_ms": self.last_ms,
        }

    def estimate(self, frame: np.ndarray, face: Face,
                 mode: str) -> AttributeSample | None:
        if self.rknn is None:
            return None
        x1, y1, x2, y2 = map(float, face["box"])
        if x2 - x1 < 1 or y2 - y1 < 1:
            return None
        aligned = crop_face(frame, face, mode, AGE_INPUT_SIZE)
        blob = cv2.dnn.blobFromImage(
            aligned, scalefactor=1.0, size=(AGE_INPUT_SIZE, AGE_INPUT_SIZE),
            mean=(0, 0, 0), swapRB=True,
        )
        blob = np.ascontiguousarray(blob, dtype=np.float32)
        started = time.perf_counter()
        try:
            nhwc = np.ascontiguousarray(
                np.transpose(blob, (0, 2, 3, 1)), dtype=np.float32)
            try:
                outputs = rknn_infer(self.rknn, nhwc, "nhwc", "float32")
            except ImportError:
                outputs = self.rknn.inference(inputs=[nhwc])
            if not outputs:
                raise RuntimeError("RKNN returned no outputs")
            values = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
            if values.size != 3:
                raise RuntimeError(
                    f"expected 3 age/gender outputs, got {values.shape}")
            self.runs += 1
            return AttributeSample(
                raw_age=float(values[2]) * 100.0,
                gender_logits=(float(values[0]), float(values[1])),
                align=mode,
                sharpness=float(face.get("sharpness", 0.0)),
            )
        except Exception as exc:
            print(f"[age-gender] inference failed: {exc}")
            return None
        finally:
            self.last_ms = round((time.perf_counter() - started) * 1000, 1)
