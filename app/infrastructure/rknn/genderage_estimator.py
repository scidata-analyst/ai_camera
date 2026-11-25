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
    FAIRFACE_AGE_BINS,
    FAIRFACE_MEAN,
    FAIRFACE_STD,
    LEGACY_AGE_INPUT_SIZE,
    MIVOLO_AGE_AVG,
    MIVOLO_AGE_MAX,
    MIVOLO_AGE_MIN,
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
        path = str(self.rknn_path).lower()
        self.mivolo = "mivolo" in path
        self.fairface = "fairface" in path and not self.mivolo
        self.input_size = (LEGACY_AGE_INPUT_SIZE
                           if not (self.fairface or self.mivolo)
                           else AGE_INPUT_SIZE)
        self._mean = np.asarray(FAIRFACE_MEAN, dtype=np.float32)
        self._std = np.asarray(FAIRFACE_STD, dtype=np.float32)
        self._age_mean = np.asarray(DLIB_AGE_MEAN, dtype=np.float32)
        self.age_rknn = None
        self.age_rknn_path = settings.age_only_rknn_path
        self.age_runs = 0
        self.age_last_ms = 0.0
        if self.rknn is not None and self.fairface and self.age_rknn_path:
            self.age_rknn, age_core = rknn_load(self.age_rknn_path, "age")
            print(f"[age] RKNN NPU active: {self.age_rknn_path} core={age_core}")

        self.runs = 0
        self.last_ms = 0.0

    def describe(self) -> dict:
        return {
            "backend": "rknn" if self.rknn is not None else "off",
            "onnx_model": self.onnx_path,
            "rknn_model": self.rknn_path,
            "model_kind": ("mivolo" if self.mivolo else
                           "fairface" if self.fairface else "genderage"),
            "input_size": self.input_size,
            "age_rknn_model": self.age_rknn_path,
            "age_loaded": (self.rknn is not None
                           and (self.age_rknn is not None or self.mivolo)),
            "age_source": ("mivolo" if self.mivolo else
                           "dlib" if self.age_rknn is not None else "fairface"),
            "age_runs": self.age_runs,
            "age_last_ms": self.age_last_ms,
            "loaded": self.rknn is not None,
            "runs": self.runs,
            "last_ms": self.last_ms,
        }

    def _blob(self, aligned: np.ndarray) -> np.ndarray:
        if self.mivolo:
            if aligned.shape[:2] != (self.input_size, self.input_size):
                scale = min(self.input_size / aligned.shape[0],
                            self.input_size / aligned.shape[1])
                size = (round(aligned.shape[1] * scale),
                        round(aligned.shape[0] * scale))
                resized = cv2.resize(
                    aligned, size, interpolation=cv2.INTER_LINEAR)
                canvas = np.zeros(
                    (self.input_size, self.input_size, 3), dtype=np.uint8)
                x = (self.input_size - size[0]) // 2
                y = (self.input_size - size[1]) // 2
                canvas[y:y + size[1], x:x + size[0]] = resized
                aligned = canvas

            rgb = aligned[:, :, ::-1]
            return np.ascontiguousarray(rgb[np.newaxis, ...], dtype=np.uint8)

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

    def _age_blob(self, aligned: np.ndarray) -> np.ndarray:
        chip = cv2.resize(
            aligned, (AGE_ONLY_INPUT_SIZE, AGE_ONLY_INPUT_SIZE),
            interpolation=cv2.INTER_AREA)
        rgb = chip[:, :, ::-1].astype(np.float32)
        rgb = (rgb - self._age_mean) / DLIB_AGE_SCALE
        return np.ascontiguousarray(
            rgb.transpose(2, 0, 1)[np.newaxis, ...], dtype=np.float32)

    def _age_value(self, outputs) -> float:
        for output in outputs:
            array = np.asarray(output, dtype=np.float32).reshape(-1)
            if array.size == DLIB_AGE_WEIGHTS.size:
                return float(np.dot(array, DLIB_AGE_WEIGHTS))

        raise RuntimeError(
            f"expected {DLIB_AGE_WEIGHTS.size} age classes, got "
            f"{[np.asarray(value).shape for value in outputs]}")

    def _estimate_age(self, aligned: np.ndarray, fallback: float) -> float:
        started = time.perf_counter()
        try:
            blob = self._age_blob(aligned)
            outputs = rknn_infer(self.age_rknn, blob, "nchw", "float32")
            if not outputs:
                raise RuntimeError("age RKNN returned no outputs")

            self.age_runs += 1
            return self._age_value(outputs)
        except Exception as exc:
            print(f"[age] inference failed: {exc}")
            return fallback
        finally:
            self.age_last_ms = round((time.perf_counter() - started) * 1000, 1)

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

    def _mivolo_sample(self, outputs, mode: str,
                       sharpness: float) -> AttributeSample:
        values = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        if values.size != 3:
            raise RuntimeError(
                f"expected 3 MiVOLO outputs, got {values.shape}")

        age = (float(values[2]) * (MIVOLO_AGE_MAX - MIVOLO_AGE_MIN)
               + MIVOLO_AGE_AVG)
        age = min(max(age, MIVOLO_AGE_MIN), MIVOLO_AGE_MAX)
        return AttributeSample(
            raw_age=age,
            gender_logits=(float(values[1]), float(values[0])),
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
        layout = "nhwc" if (self.mivolo or not self.fairface) else "nchw"
        dtype = None if self.mivolo else "float32"
        started = time.perf_counter()
        try:
            outputs = rknn_infer(self.rknn, blob, layout, dtype)
            if not outputs:
                raise RuntimeError("RKNN returned no outputs")

            self.runs += 1
            if self.mivolo:
                return self._mivolo_sample(
                    outputs, mode, float(face.get("sharpness", 0.0)))

            if self.fairface:
                sample = self._fairface_sample(
                    outputs, mode, float(face.get("sharpness", 0.0)))
                if self.age_rknn is not None:
                    age = self._estimate_age(aligned, sample.raw_age)
                    sample = AttributeSample(
                        raw_age=age,
                        gender_logits=sample.gender_logits,
                        align=sample.align,
                        sharpness=sample.sharpness,
                    )

                return sample

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
