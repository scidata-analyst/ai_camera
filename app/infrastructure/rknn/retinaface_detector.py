from __future__ import annotations

import cv2
import numpy as np

from ...config.constants import FACE_DETECTOR_SIZE
from ...config.settings import Settings
from ...domain.geometry import (
    decode_boxes,
    decode_landmarks,
    face_letterbox,
    nms,
    prior_boxes,
)
from ...domain.face.ports import FaceDetector
from ...domain.models import Face
from .runtime import rknn_infer, rknn_load

_PRIORS = prior_boxes(FACE_DETECTOR_SIZE)


class RknnFaceDetector(FaceDetector):
    def __init__(self, settings: Settings):
        self.score_thr = settings.face_score
        if settings.face_backend == "off":
            self.rknn = None
            self.rknn_path = settings.face_rknn_path
            return
        if settings.face_backend not in ("rknn", ""):
            raise RuntimeError(
                f"FACE_BACKEND={settings.face_backend!r} is disabled in "
                "NPU-only mode")
        self.rknn_path = settings.face_rknn_path
        self.rknn, core = rknn_load(self.rknn_path, "face")
        print(f"[face] RetinaFace RKNN NPU active: {self.rknn_path} core={core}")

    def describe(self) -> dict:
        return {
            "backend": "rknn" if self.rknn is not None else "off",
            "rknn_model": self.rknn_path,
            "loaded": self.rknn is not None,
        }

    def detect(self, frame: np.ndarray) -> list[Face]:
        if self.rknn is None:
            return []
        height, width = frame.shape[:2]
        image, ratio, pad_x, pad_y = face_letterbox(frame)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        blob = np.ascontiguousarray(rgb[np.newaxis, ...], dtype=np.uint8)
        try:
            try:
                outputs = rknn_infer(self.rknn, blob, "nhwc", "uint8")
            except ImportError:
                outputs = self.rknn.inference(inputs=[blob])
            if not outputs:
                raise RuntimeError("RKNN returned no outputs")
            if len(outputs) != 3:
                raise RuntimeError(
                    f"expected loc/conf/landmark outputs, got {len(outputs)}")
            loc, conf, raw_landmarks = [
                np.asarray(output, dtype=np.float32).squeeze(0)
                for output in outputs]
            if loc.shape != (len(_PRIORS), 4) or conf.shape != (len(_PRIORS), 2):
                raise RuntimeError(
                    f"unexpected RetinaFace outputs: loc={loc.shape}, "
                    f"conf={conf.shape}")

            boxes = decode_boxes(_PRIORS, loc) * FACE_DETECTOR_SIZE
            landmarks = decode_landmarks(_PRIORS, raw_landmarks) * FACE_DETECTOR_SIZE
            scores = conf[:, 1]
            selected = np.flatnonzero(scores >= self.score_thr)
            if selected.size == 0:
                return []
            boxes, scores, landmarks = (boxes[selected], scores[selected],
                                        landmarks[selected])
            boxes[:, 0::2] = np.clip((boxes[:, 0::2] - pad_x) / ratio, 0, width)
            boxes[:, 1::2] = np.clip((boxes[:, 1::2] - pad_y) / ratio, 0, height)
            points = landmarks.reshape(-1, 5, 2)
            points[:, :, 0] = np.clip((points[:, :, 0] - pad_x) / ratio,
                                      0, width - 1)
            points[:, :, 1] = np.clip((points[:, :, 1] - pad_y) / ratio,
                                      0, height - 1)

            order = scores.argsort()[::-1]
            boxes, scores, points = boxes[order], scores[order], points[order]
            kept = nms(boxes, scores, 0.5)
            return [{
                "box": [int(round(value)) for value in boxes[index]],
                "conf": round(float(scores[index]), 3),
                "landmarks": np.rint(points[index]).astype(int).tolist(),
            } for index in kept]
        except Exception as exc:
            print(f"[face] RetinaFace inference failed: {exc}")
            return []
