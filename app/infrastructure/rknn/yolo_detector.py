from __future__ import annotations

import os
import threading

import cv2
import numpy as np

from ...config.constants import PERSON_CLASS_NAME
from ...config.settings import Settings
from ...domain.detection.ports import PersonDetector
from ...domain.geometry import letterbox
from ...domain.models import Detection
from .runtime import hw_summary, rknn_infer, rknn_input_summary, rknn_load


class RknnPersonDetector(PersonDetector):
    def __init__(self, settings: Settings):

        if settings.yolo_backend not in ("rknn", ""):
            raise RuntimeError(
                f"YOLO_BACKEND={settings.yolo_backend!r} is disabled in "
                "NPU-only mode")
        self.rknn_path   = settings.rknn_model
        self.imgsz       = settings.imgsz
        self.conf        = settings.conf
        self.person_conf = settings.person_conf
        self.debug       = settings.yolo_debug
        self._lock       = threading.Lock()
        self._warned     = False
        self._debug_done = False

        try:
            print(f"[hw] {hw_summary()} rknn_model={self.rknn_path} "
                  f"exists={os.path.exists(self.rknn_path)}")
        except Exception:
            pass
        self.rknn, _ = rknn_load(self.rknn_path, "yolo")
        print(f"[yolo] NPU input: {rknn_input_summary(self.rknn)}")
        print(f"[yolo] NPU backend active: {self.rknn_path}")

    def describe(self) -> dict:
        return {"backend": "rknn", "rknn_model": self.rknn_path, "npu_active": self.rknn is not None, "loaded": self.rknn is not None, "input": rknn_input_summary(self.rknn),}

    def _empty(self):

        if not self._warned:
            print("[yolo] inference returned no detections")
            self._warned = True
        return []

    def detect(self, frame: np.ndarray) -> list[Detection]:

        if self.rknn is None:
            raise RuntimeError("[yolo] NPU backend unavailable — CPU disabled")

        try:
            return self._detect_boxes(frame)
        except Exception as exc:

            if not self._warned:
                print(f"[yolo] inference failed: {exc}")
                self._warned = True
            return []

    def _detect_boxes(self, frame: np.ndarray) -> list[Detection]:

        if self.rknn is None:
            raise RuntimeError("[yolo] NPU backend unavailable — CPU disabled")
        height, width = frame.shape[:2]
        image, ratio, pad_x, pad_y = letterbox(frame, self.imgsz)
        rgb  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(rgb[np.newaxis, ...], dtype=np.float32)
        with self._lock:
            outputs = rknn_infer(self.rknn, blob, "nhwc", "float32")

        if not outputs:
            return self._empty()

        prediction = None

        for output in outputs:
            array = np.asarray(output)

            if array.ndim == 3 and (array.shape[1] == 84 or array.shape[2] == 84):
                prediction = array
                break

        if prediction is None:

            if not self._warned:
                print(f"[yolo] NPU unexpected outputs: "
                      f"{[tuple(np.asarray(o).shape) for o in outputs]} "
                      f"-> empty (shape janiye pathao, fix kore debo)")
                self._warned = True
            return []

        if prediction.shape[1] == 84:
            prediction = np.transpose(prediction[0], (1, 0))
        else:
            prediction = prediction[0]

        boxes_cxcywh  = prediction[:, :4]
        scores        = prediction[:, 4:]
        person_scores = scores[:, 0]

        if self.debug and not self._debug_done:
            self._debug_done = True
            best             = float(person_scores.max()) if len(person_scores) else 0.0
            print(f"[yolo-debug] max person score={best:.3f}")

        keep         = person_scores >= self.person_conf
        boxes_cxcywh = boxes_cxcywh[keep]
        confs        = person_scores[keep]

        if len(confs) == 0:
            return []

        cx, cy, bw, bh = boxes_cxcywh.T
        x1        = ((cx - bw / 2 - pad_x) / ratio).clip(0, width)
        y1        = ((cy - bh / 2 - pad_y) / ratio).clip(0, height)
        x2        = ((cx + bw / 2 - pad_x) / ratio).clip(0, width)
        y2        = ((cy + bh / 2 - pad_y) / ratio).clip(0, height)
        corners   = np.stack([x1, y1, x2, y2], axis=1).tolist()
        nms_boxes = [[int(box[0]), int(box[1]), max(1, int(box[2] - box[0])), max(1, int(box[3] - box[1]))] for box in corners]

        try:
            picked = cv2.dnn.NMSBoxes(nms_boxes, confs.tolist(), self.conf, 0.6)
            picked = [int(value) for value in np.asarray(picked).reshape(-1)]
        except Exception:
            picked = list(range(len(nms_boxes)))
        picked = sorted(picked, key=lambda index: -float(confs[index]))
        return[{"kind": "human", "class": PERSON_CLASS_NAME, "conf": round(float(confs[index]), 3), "box": [int(corners[index][0]), int(corners[index][1]), int(corners[index][2]), int(corners[index][3])],} for index in picked]
