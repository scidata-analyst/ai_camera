"""InsightFace's lightweight age/gender head with RKNN and ONNX CPU backends."""
from __future__ import annotations

import math
import os
import time

import cv2
import numpy as np


INPUT_SIZE = 96
DEFAULT_ONNX = "/app/models/genderage.onnx"
DEFAULT_RKNN = "/app/models/genderage-rk3576.rknn"

# InsightFace arcface 5-point template, normalised to a 112x112 crop.
_ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32) / 112.0


class AgeGender:
    def __init__(self):
        self.onnx_path = os.getenv("AGE_GENDER_MODEL", DEFAULT_ONNX)
        self.rknn_path = os.getenv("AGE_GENDER_RKNN_MODEL", DEFAULT_RKNN)
        self.backend_want = os.getenv("AGE_GENDER_BACKEND", "rknn").lower()
        # Quality gates: skip blurry crops, choose crop alignment, calibrate age.
        self.blur_thr = float(os.getenv("FACE_BLUR", "80"))
        self.align_mode = os.getenv("AGE_GENDER_ALIGN", "roll").lower()
        if self.align_mode not in ("box", "roll", "arcface"):
            self.align_mode = "roll"
        self.age_scale = float(os.getenv("AGE_SCALE", "1.0"))
        self.age_bias = float(os.getenv("AGE_BIAS", "0.0"))
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
    def _sharpness(frame: np.ndarray, box) -> float:
        """Laplacian variance of the raw face crop; low = motion blur."""
        x1, y1, x2, y2 = map(int, box)
        h, w = frame.shape[:2]
        xa, ya, xb, yb = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if xb - xa < 2 or yb - ya < 2:
            return 0.0
        gray = cv2.cvtColor(frame[ya:yb, xa:xb], cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _landmarks(face: dict) -> np.ndarray | None:
        lm = face.get("landmarks")
        if lm is None or len(lm) != 5:
            return None
        try:
            src = np.asarray(lm, dtype=np.float32).reshape(5, 2)
        except (TypeError, ValueError):
            return None
        return src if np.all(np.isfinite(src)) else None

    @classmethod
    def _crop(cls, frame: np.ndarray, face: dict, mode: str) -> np.ndarray:
        x1, y1, x2, y2 = map(float, face["box"])
        width, height = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        scale = INPUT_SIZE / (max(width, height) * 1.5)
        src = cls._landmarks(face)

        if mode == "arcface" and src is not None:
            # Full 5-point arcface alignment (experimental: this model was
            # trained on box-centre crops, so 'roll' is the safer default).
            ratio = INPUT_SIZE / 112.0
            dst = _ARCFACE_DST * ratio
            diff = 8.0 * ratio
            dst[:, 0] += diff
            dst[:, 1] += diff
            matrix, _ = cv2.estimateAffinePartial2D(src, dst)
            if matrix is not None:
                return cv2.warpAffine(
                    frame, matrix, (INPUT_SIZE, INPUT_SIZE), borderValue=0.0)

        if mode == "roll" and src is not None:
            # Keep the official 1.5x framing but level the eye line, so a
            # tilted head does not change the predicted age/gender.
            le, re = src[0], src[1]
            if abs(re[0] - le[0]) > 1.0:
                roll = math.degrees(math.atan2(
                    float(re[1] - le[1]), float(re[0] - le[0])))
                roll = max(-30.0, min(30.0, roll))
                if abs(roll) > 1.0:
                    matrix = cv2.getRotationMatrix2D((cx, cy), roll, scale)
                    matrix[0, 2] += INPUT_SIZE / 2.0 - cx
                    matrix[1, 2] += INPUT_SIZE / 2.0 - cy
                    return cv2.warpAffine(
                        frame, matrix, (INPUT_SIZE, INPUT_SIZE),
                        borderValue=0.0)

        # Default / fallback: InsightFace Attribute.get box-centre transform.
        matrix = np.asarray([
            [scale, 0.0, INPUT_SIZE / 2.0 - cx * scale],
            [0.0, scale, INPUT_SIZE / 2.0 - cy * scale],
        ], dtype=np.float32)
        return cv2.warpAffine(
            frame, matrix, (INPUT_SIZE, INPUT_SIZE), borderValue=0.0)

    @classmethod
    def _input(cls, frame: np.ndarray, face: dict,
               mode: str) -> np.ndarray | None:
        x1, y1, x2, y2 = map(float, face["box"])
        if x2 - x1 < 1 or y2 - y1 < 1:
            return None
        aligned = cls._crop(frame, face, mode)
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
        sharp = self._sharpness(frame, face["box"])
        face["sharpness"] = round(sharp, 1)
        # Motion blur: no fresh guess. The smoother keeps this person's last
        # value instead of locking a wrong age/gender from a blurred crop.
        if self.blur_thr > 0 and sharp < self.blur_thr:
            face.update({"gender": "Unknown", "gconf": 0.0, "age": 0,
                         "blurry": True})
            return face
        face["blurry"] = False
        blob = self._input(frame, face, self.align_mode)
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
            raw_age = float(values[2]) * 100.0
            # Optional linear correction: AGE_SCALE * raw + AGE_BIAS.
            age = int(np.clip(np.rint(raw_age * self.age_scale + self.age_bias),
                              0, 120))
            face.update({
                "gender": gender,
                "gconf": round(float(probs[gender_id]), 3),
                "age": age,
                "raw_age": round(raw_age, 1),
                "align": self.align_mode,
            })
            self.runs += 1
            return face
        except Exception as e:
            print(f"[age-gender] inference failed: {e}")
            face.update({"gender": "Unknown", "gconf": 0.0, "age": 0})
            return face
        finally:
            self.last_ms = round((time.perf_counter() - started) * 1000, 1)
