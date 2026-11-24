from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from ...config.constants import WHENET_INPUT_SIZE
from ...config.settings import Settings
from ...domain.geometry import iou
from ...domain.headpose.ports import HeadPoseAnalyzer
from ...domain.models import Face
from .runtime import rknn_infer, rknn_load


class RknnHeadPoseEstimator(HeadPoseAnalyzer):
    def __init__(self, settings: Settings):
        self.enabled = settings.headpose_enable
        self.yaw_thr = settings.look_yaw
        self.pitch_thr = settings.look_pitch
        self.rknn_path = settings.headpose_rknn_path
        self.onnx_path = settings.headpose_onnx_path
        self.rknn = None
        self.backend = "off"
        self.providers = ["CPUExecutionProvider"]
        self._clock = threading.RLock()
        self._cache = []
        self._ttl = settings.headpose_ttl
        self._iou_thr = 0.4
        if not self.enabled or settings.headpose_backend == "off":
            return
        if settings.headpose_backend not in ("rknn", ""):
            raise RuntimeError(
                f"HEADPOSE_BACKEND={settings.headpose_backend!r} is disabled "
                "in NPU-only mode")
        self.rknn, core = rknn_load(self.rknn_path, "headpose")
        self.backend = "rknn"
        self.providers = ["RKNN NPU"]
        print(f"[headpose] WHENet RKNN NPU active: "
              f"{self.rknn_path} core={core}")

    def describe(self) -> dict:
        return {
            "backend": self.backend,
            "rknn_model": self.rknn_path,
            "providers": list(self.providers),
            "loaded": self.rknn is not None,
        }

    def _infer(self, batch: np.ndarray) -> np.ndarray:
        nhwc = np.ascontiguousarray(
            np.transpose(batch, (0, 2, 3, 1)), dtype=np.float32)
        rows = []
        for sample in nhwc:
            blob = sample[np.newaxis, ...]
            try:
                outputs = rknn_infer(self.rknn, blob, "nhwc", "float32")
            except ImportError:
                outputs = self.rknn.inference(inputs=[blob])
            if not outputs:
                raise RuntimeError("RKNN returned no outputs")
            rows.append(np.asarray(outputs[0], dtype=np.float32).reshape(-1)[:3])
        return np.asarray(rows, dtype=np.float32)

    @staticmethod
    def _preprocess(crop: np.ndarray) -> np.ndarray:
        image = cv2.resize(crop, (WHENET_INPUT_SIZE, WHENET_INPUT_SIZE))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return np.transpose(image, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

    def predict_crop(self, crop: np.ndarray):
        output = self._infer(self._preprocess(crop))[0]
        yaw, roll, pitch = float(output[0]), float(output[1]), float(output[2])
        return yaw, pitch, roll

    def remember(self, faces: list[Face]) -> list[Face]:
        now = time.time()
        with self._clock:
            for face in faces:
                if "looking" not in face:
                    continue
                entry = {
                    "box": list(face["box"]),
                    "yaw": face.get("yaw", 0.0),
                    "pitch": face.get("pitch", 0.0),
                    "roll": face.get("roll", 0.0),
                    "looking": bool(face.get("looking")),
                    "t": now,
                }
                replaced = False
                for cached in self._cache:
                    if iou(face["box"], cached["box"]) >= self._iou_thr:
                        cached.update(entry)
                        replaced = True
                        break
                if not replaced:
                    self._cache.append(entry)
            self._cache = self._cache[-20:]
        return faces

    def fill_from_cache(self, faces: list[Face]) -> list[Face]:
        if self.rknn is None:
            return faces
        now = time.time()
        with self._clock:
            for face in faces:
                best, best_iou = None, 0.0
                for cached in self._cache:
                    overlap = iou(face["box"], cached["box"])
                    if overlap > best_iou:
                        best, best_iou = cached, overlap
                if best is not None and best_iou >= self._iou_thr \
                        and now - best["t"] <= self._ttl:
                    best["box"] = list(face["box"])
                    face.update({
                        "yaw": best["yaw"],
                        "pitch": best["pitch"],
                        "roll": best["roll"],
                        "looking": best["looking"],
                    })
        return faces

    def predict_frame(self, frame: np.ndarray, faces: list[Face],
                      min_side: int = 40) -> list[Face]:
        if self.rknn is None:
            return faces
        height, width = frame.shape[:2]
        blobs, indexes = [], []
        for index, face in enumerate(faces):
            x1, y1, x2, y2 = face["box"]
            box_width, box_height = x2 - x1, y2 - y1
            if min(box_width, box_height) < min_side:
                face.update({"yaw": 0.0, "pitch": 0.0, "roll": 0.0,
                             "looking": False})
                continue
            xa = max(int(x1 - box_width / 5), 0)
            xb = min(int(x2 + box_width / 5), width)
            ya = max(int(y1 - box_height / 10), 0)
            yb = min(int(y2 + box_height / 10), height)
            crop = frame[ya:yb, xa:xb]
            if crop.size == 0:
                face.update({"yaw": 0.0, "pitch": 0.0, "roll": 0.0,
                             "looking": False})
                continue
            try:
                image = cv2.resize(crop, (WHENET_INPUT_SIZE, WHENET_INPUT_SIZE))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                blobs.append(np.transpose(image, (2, 0, 1)))
                indexes.append(index)
            except Exception as exc:
                print(f"[headpose] preprocess fail: {exc}")
                face.update({"yaw": 0.0, "pitch": 0.0, "roll": 0.0,
                             "looking": False})
        if blobs:
            try:
                batch = np.stack(blobs, axis=0).astype(np.float32)
                output = self._infer(batch)
                for row, index in zip(output, indexes):
                    yaw, roll, pitch = float(row[0]), float(row[1]), float(row[2])
                    looking = bool(abs(yaw) <= self.yaw_thr
                                   and abs(pitch) <= self.pitch_thr)
                    faces[index].update({
                        "yaw": round(yaw, 1),
                        "pitch": round(pitch, 1),
                        "roll": round(roll, 1),
                        "looking": looking,
                    })
            except Exception as exc:
                print(f"[headpose] predict fail: {exc}")
                for index in indexes:
                    faces[index].update({"yaw": 0.0, "pitch": 0.0,
                                         "roll": 0.0, "looking": False})
        return faces
