from __future__ import annotations

import time

import cv2
import numpy as np

from ...config.constants import (
    AGE_INPUT_SIZE,
    FAIRFACE_AGE_BINS,
    FAIRFACE_MEAN,
    FAIRFACE_STD,
    LEGACY_AGE_INPUT_SIZE,
)
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
        self.fairface = "fairface" in str(self.rknn_path).lower()
        self.input_size = AGE_INPUT_SIZE if self.fairface else LEGACY_AGE_INPUT_SIZE
        self._mean = np.asarray(FAIRFACE_MEAN, dtype=np.float32)
        self._std = np.asarray(FAIRFACE_STD, dtype=np.float32)
        self.runs = 0
        self.last_ms = 0.0

    def describe(self) -> dict:
        return {
            "backend": "rknn" if self.rknn is not None else "off",
            "onnx_model": self.onnx_path,
            "rknn_model": self.rknn_path,
            "model_kind": "fairface" if self.fairface else "genderage",
            "input_size": self.input_size,
            "loaded": self.rknn is not None,
            "runs": self.runs,
            "last_ms": self.last_ms,
        }

    def _blob(self, aligned: np.ndarray) -> np.ndarray:
        if self.fairface:
            rgb = aligned[:, :, ::-1].astype(np.float32) / 255.0
            rgb = (rgb - self._mean) / self._std
            return np.ascontiguousarray(
                rgb.transpose(2, 0, 1)[np.newaxis, ...], dtype=np.float32)
        blob = cv2.dnn.blobFromImage(
            aligned, scalefactor=1.0,
            size=(self.input_size, self.input_size),
            mean=(0, 0, 0), swapRB=True,
        )
        return np.ascontiguousarray(
            np.transpose(blob, (0, 2, 3, 1)), dtype=np.float32)

    def _output_names(self) -> list[str]:
        try:
            details = self.rknn.get_output_details() or []
        except Exception:
            return []
        return [str(detail.get("name", "")).lower() for detail in details]

    def _fairface_sample(self, outputs, mode: str,
                         sharpness: float) -> AttributeSample:
        arrays = [np.asarray(value, dtype=np.float32).reshape(-1)
                  for value in outputs]
        names = self._output_names()
        gender = None
        age = None
        age_bins = None
        for index, array in enumerate(arrays):
            name = names[index] if index < len(names) else ""
            if gender is None and ("gender" in name or array.size == 2):
                gender = array
            elif age_bins is None and ("age_prob" in name or array.size == 9):
                age_bins = array
            elif age is None and ("age_value" in name or array.size == 1):
                age = float(array[0])
        if gender is None and len(arrays) > 1:
            gender = arrays[1]
        if age is None and age_bins is not None:
            age = float(np.dot(age_bins, FAIRFACE_AGE_BINS))
        if age is None and len(arrays) > 2:
            age = float(arrays[2][0])
        if gender is None or gender.size < 2 or age is None:
            raise RuntimeError(
                f"expected FairFace gender/age outputs, got "
                f"{[array.shape for array in arrays]}")
        return AttributeSample(
            raw_age=age,
            gender_logits=(float(gender[1]), float(gender[0])),
            align=mode,
            sharpness=sharpness,
        )

    def estimate(self, frame: np.ndarray, face: Face,
                 mode: str) -> AttributeSample | None:
        if self.rknn is None:
            return None
        x1, y1, x2, y2 = map(float, face["box"])
        if x2 - x1 < 1 or y2 - y1 < 1:
            return None
        aligned = crop_face(frame, face, mode, self.input_size)
        blob = self._blob(aligned)
        layout = "nchw" if self.fairface else "nhwc"
        started = time.perf_counter()
        try:
            outputs = rknn_infer(self.rknn, blob, layout, "float32")
            if not outputs:
                raise RuntimeError("RKNN returned no outputs")
            self.runs += 1
            if self.fairface:
                return self._fairface_sample(
                    outputs, mode, float(face.get("sharpness", 0.0)))
            values = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
            if values.size != 3:
                raise RuntimeError(
                    f"expected 3 age/gender outputs, got {values.shape}")
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
