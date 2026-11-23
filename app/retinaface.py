"""RetinaFace 320x320 face detector with RKNN and OpenCV ONNX backends."""
from __future__ import annotations

import math
import os
from itertools import product

import cv2
import numpy as np


MODEL_SIZE = 320
DEFAULT_ONNX = "/app/models/RetinaFace_mobile320.onnx"
DEFAULT_RKNN = "/app/models/RetinaFace_mobile320-rk3576.rknn"


def _prior_boxes(size: int = MODEL_SIZE) -> np.ndarray:
    anchors = []
    min_sizes = ((16, 32), (64, 128), (256, 512))
    steps = (8, 16, 32)
    for level, step in enumerate(steps):
        rows = math.ceil(size / step)
        cols = math.ceil(size / step)
        for row, col in product(range(rows), range(cols)):
            for min_size in min_sizes[level]:
                anchors.append(((col + 0.5) * step / size,
                                (row + 0.5) * step / size,
                                min_size / size, min_size / size))
    return np.asarray(anchors, dtype=np.float32)


_PRIORS = _prior_boxes()


def _decode_boxes(loc: np.ndarray) -> np.ndarray:
    variances = np.asarray((0.1, 0.2), dtype=np.float32)
    boxes = np.concatenate((
        _PRIORS[:, :2] + loc[:, :2] * variances[0] * _PRIORS[:, 2:],
        _PRIORS[:, 2:] * np.exp(loc[:, 2:] * variances[1]),
    ), axis=1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def _decode_landmarks(raw: np.ndarray) -> np.ndarray:
    points = []
    for i in range(5):
        points.append(_PRIORS[:, :2] + raw[:, 2 * i:2 * i + 2] *
                      0.1 * _PRIORS[:, 2:])
    return np.concatenate(points, axis=1)


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)
    order = scores.argsort()[::-1]
    kept: list[int] = []
    while order.size:
        i = int(order[0])
        kept.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-6)
        order = rest[iou <= threshold]
    return kept


class RetinaFace:
    def __init__(self, score: float = 0.6):
        self.score_thr = float(os.getenv("FACE_SCORE", str(score)))
        self.backend_want = os.getenv("FACE_BACKEND", "rknn").lower()
        self.onnx_path = os.getenv("FACE_MODEL", DEFAULT_ONNX)
        self.rknn_path = os.getenv("FACE_RKNN_MODEL", DEFAULT_RKNN)
        self.rknn = None
        self.net = None
        self.backend_used = "off"

        if self.backend_want == "off":
            return
        if self.backend_want not in ("rknn", ""):
            raise RuntimeError(
                f"FACE_BACKEND={self.backend_want!r} is disabled in NPU-only mode")
        from .accel import rknn_load
        self.rknn, core = rknn_load(self.rknn_path, "face")
        self.backend_used = "rknn"
        print(f"[face] RetinaFace RKNN NPU active: {self.rknn_path} core={core}")

    @property
    def loaded(self) -> bool:
        return self.rknn is not None or self.net is not None

    @staticmethod
    def _letterbox(frame: np.ndarray):
        h, w = frame.shape[:2]
        ratio = min(MODEL_SIZE / w, MODEL_SIZE / h)
        nw, nh = int(w * ratio), int(h * ratio)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        pad_x, pad_y = (MODEL_SIZE - nw) // 2, (MODEL_SIZE - nh) // 2
        canvas = np.full((MODEL_SIZE, MODEL_SIZE, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
        return canvas, ratio, pad_x, pad_y

    def detect(self, frame: np.ndarray) -> list[dict]:
        if not self.loaded:
            return []
        h, w = frame.shape[:2]
        letterboxed, ratio, pad_x, pad_y = self._letterbox(frame)
        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        # RKNN needs an explicit batch: (1, 320, 320, 3), not a bare HWC array.
        rknn_blob = np.ascontiguousarray(rgb[np.newaxis, ...], dtype=np.uint8)
        try:
            if self.rknn is not None:
                try:
                    from .accel import rknn_infer
                    outputs = rknn_infer(self.rknn, rknn_blob, "nhwc", "uint8")
                except ImportError:
                    outputs = self.rknn.inference(inputs=[rknn_blob])
                if not outputs:
                    raise RuntimeError("RKNN returned no outputs")
            else:
                # RKNN conversion applies the same RGB mean subtraction.
                blob = cv2.dnn.blobFromImage(
                    rgb, scalefactor=1.0, size=(MODEL_SIZE, MODEL_SIZE),
                    mean=(104, 117, 123), swapRB=False,
                )
                self.net.setInput(blob)
                outputs = self.net.forward(self.output_names)
            if len(outputs) != 3:
                raise RuntimeError(f"expected loc/conf/landmark outputs, got {len(outputs)}")
            loc, conf, raw_landmarks = [np.asarray(x, dtype=np.float32).squeeze(0)
                                        for x in outputs]
            if loc.shape != (len(_PRIORS), 4) or conf.shape != (len(_PRIORS), 2):
                raise RuntimeError(
                    f"unexpected RetinaFace outputs: loc={loc.shape}, conf={conf.shape}")

            boxes = _decode_boxes(loc) * MODEL_SIZE
            landmarks = _decode_landmarks(raw_landmarks) * MODEL_SIZE
            scores = conf[:, 1]
            selected = np.flatnonzero(scores >= self.score_thr)
            if selected.size == 0:
                return []
            boxes, scores, landmarks = boxes[selected], scores[selected], landmarks[selected]
            boxes[:, 0::2] = np.clip((boxes[:, 0::2] - pad_x) / ratio, 0, w)
            boxes[:, 1::2] = np.clip((boxes[:, 1::2] - pad_y) / ratio, 0, h)
            lm = landmarks.reshape(-1, 5, 2)
            lm[:, :, 0] = np.clip((lm[:, :, 0] - pad_x) / ratio, 0, w - 1)
            lm[:, :, 1] = np.clip((lm[:, :, 1] - pad_y) / ratio, 0, h - 1)

            # Kono count cap nai — joto face score pass korbe sob NMS-e jay.
            order = scores.argsort()[::-1]
            boxes, scores, lm = boxes[order], scores[order], lm[order]
            keep = _nms(boxes, scores, 0.5)
            return [{
                "box": [int(round(v)) for v in boxes[i]],
                "conf": round(float(scores[i]), 3),
                "landmarks": np.rint(lm[i]).astype(int).tolist(),
            } for i in keep]
        except Exception as e:
            print(f"[face] RetinaFace inference failed: {e}")
            return []

    @staticmethod
    def draw(frame: np.ndarray, faces) -> np.ndarray:
        for face in faces:
            x1, y1, x2, y2 = face["box"]
            color = (0, 255, 0) if face.get("looking") else (255, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            for px, py in face.get("landmarks", []):
                cv2.circle(frame, (int(px), int(py)), 2, (0, 0, 255), -1)
            if face.get("gender") not in (None, "Unknown"):
                label = f"{face['gender']} {face['gconf']:.2f} {face['age']}y"
                cv2.putText(frame, label, (x1, max(y1 - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)
        return frame
