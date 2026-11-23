"""InsightFace's lightweight age/gender head with RKNN and ONNX CPU backends."""
from __future__ import annotations

import os
import time

import cv2
import numpy as np


INPUT_SIZE = 96
DEFAULT_ONNX = "/app/models/genderage.onnx"
DEFAULT_RKNN = "/app/models/genderage-rk3576.rknn"


class AgeGender:
    def __init__(self):
        self.onnx_path = os.getenv("AGE_GENDER_MODEL", DEFAULT_ONNX)
        self.rknn_path = os.getenv("AGE_GENDER_RKNN_MODEL", DEFAULT_RKNN)
        self.backend_want = os.getenv("AGE_GENDER_BACKEND", "rknn").lower()
        self.backend = "off"
        self.rknn = None
        self.net = None
        self.runs = 0
        self.last_ms = 0.0

        if self.backend_want == "off":
            return
        if self.backend_want not in ("rknn", ""):
            raise RuntimeError(
                f"AGE_GENDER_BACKEND={self.backend_want!r} is disabled in NPU-only mode")
        from .accel import rknn_load
        self.rknn, core = rknn_load(self.rknn_path, "age-gender")
        self.backend = "rknn"
        print(f"[age-gender] RKNN NPU active: {self.rknn_path} core={core}")

    @property
    def loaded(self) -> bool:
        return self.rknn is not None or self.net is not None

    @staticmethod
    def _input(frame: np.ndarray, box) -> np.ndarray | None:
        x1, y1, x2, y2 = map(float, box)
        width, height = x2 - x1, y2 - y1
        if width < 1 or height < 1:
            return None
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        scale = INPUT_SIZE / (max(width, height) * 1.5)
        transform = np.asarray([
            [scale, 0.0, INPUT_SIZE / 2.0 - cx * scale],
            [0.0, scale, INPUT_SIZE / 2.0 - cy * scale],
        ], dtype=np.float32)
        aligned = cv2.warpAffine(
            frame, transform, (INPUT_SIZE, INPUT_SIZE), borderValue=0.0)
        # The official ONNX graph contains its own -127.5 / 128 normalization.
        # Feed raw RGB pixels, exactly as InsightFace's Attribute.get does.
        blob = cv2.dnn.blobFromImage(
            aligned, scalefactor=1.0, size=(INPUT_SIZE, INPUT_SIZE),
            mean=(0, 0, 0), swapRB=True,
        )
        return np.ascontiguousarray(blob, dtype=np.float32)

    def predict(self, frame: np.ndarray, face: dict) -> dict:
        if not self.loaded:
            face.update({"gender": "Unknown", "gconf": 0.0, "age": 0})
            return face
        blob = self._input(frame, face["box"])
        if blob is None:
            face.update({"gender": "Unknown", "gconf": 0.0, "age": 0})
            return face
        started = time.perf_counter()
        try:
            if self.rknn is not None:
                # blob already (1,3,96,96); reorder axes on the 4D array.
                nhwc = np.ascontiguousarray(
                    np.transpose(blob, (0, 2, 3, 1)), dtype=np.float32)
                try:
                    from .accel import rknn_infer
                    outs = rknn_infer(self.rknn, nhwc, "nhwc", "float32")
                except ImportError:
                    outs = self.rknn.inference(inputs=[nhwc])
                if not outs:
                    raise RuntimeError("RKNN returned no outputs")
                raw = outs[0]
            else:
                self.net.setInput(blob)
                raw = self.net.forward()
            values = np.asarray(raw, dtype=np.float32).reshape(-1)
            if values.size != 3:
                raise RuntimeError(f"expected 3 age/gender outputs, got {values.shape}")
            logits = values[:2]
            logits = logits - np.max(logits)
            probs = np.exp(logits)
            probs /= max(float(np.sum(probs)), 1e-12)
            gender_id = int(np.argmax(probs))
            # InsightFace GenderAge: 0 = female, 1 = male; age is scaled by 100.
            gender = "Male" if gender_id == 1 else "Female"
            age = int(np.clip(np.rint(float(values[2]) * 100.0), 0, 100))
            face.update({
                "gender": gender,
                "gconf": round(float(probs[gender_id]), 3),
                "age": age,
            })
            self.runs += 1
            return face
        except Exception as e:
            print(f"[age-gender] inference failed: {e}")
            face.update({"gender": "Unknown", "gconf": 0.0, "age": 0})
            return face
        finally:
            self.last_ms = round((time.perf_counter() - started) * 1000, 1)
