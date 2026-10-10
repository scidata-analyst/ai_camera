"""YOLOv11 based Human vs Non-Human detector."""
import threading
import time
from typing import List, Dict

import cv2
import numpy as np

from .smoother import CountSmoother


class HumanDetector:
    """Wraps ultralytics YOLO. person (class 0) -> Human, else Non-Human."""

    def __init__(self, model_name: str = "yolo11n.pt", conf: float = 0.5, imgsz: int = 640,
                 window_sec: float = 1.0):
        from ultralytics import YOLO

        self.conf = conf
        self.imgsz = imgsz
        self.model_name = model_name
        # lazy download on first run; Docker build e cache kora jabe
        self.model = YOLO(model_name)
        self.names = self.model.names  # {0: 'person', ...}
        self._lock = threading.Lock()
        self.last_counts = {"human": 0, "non_human": 0, "fps": 0.0}
        self.smoother = CountSmoother(window_sec=window_sec)
        self.smoothed = self.smoother.current
        self._t_prev = time.time()
        self._fps_smooth = 0.0

    def infer(self, frame: np.ndarray) -> tuple[np.ndarray, List[Dict]]:
        """Run detection, draw boxes, return (annotated_frame, detections)."""
        with self._lock:
            results = self.model.predict(
                source=frame, conf=self.conf, imgsz=self.imgsz, verbose=False
            )[0]

        detections: List[Dict] = []
        human_n = 0
        non_human_n = 0

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            coco_name = self.names.get(cls_id, str(cls_id))

            if cls_id == 0:  # person
                label = f"Human {conf:.2f}"
                color = (0, 255, 0)  # green
                kind = "human"
                human_n += 1
            else:
                label = f"Non-Human:{coco_name} {conf:.2f}"
                color = (0, 165, 255)  # orange
                kind = "non_human"
                non_human_n += 1

            detections.append(
                {"kind": kind, "class": coco_name, "conf": round(conf, 3),
                 "box": [x1, y1, x2, y2]}
            )

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # FPS + 1-sec smoothed header (mean/median/mode)
        now = time.time()
        inst = 1.0 / max(now - self._t_prev, 1e-6)
        self._t_prev = now
        self._fps_smooth = self._fps_smooth * 0.9 + inst * 0.1

        self.smoothed = self.smoother.add(human_n, non_human_n)
        sh = self.smoothed["stable_human"]
        sn = self.smoothed["stable_non_human"]
        mh = self.smoothed["human"]["mean"]

        header = (f"Human:{human_n}[stable:{sh}] Non-Human:{non_human_n}[stable:{sn}] "
                  f"FPS:{self._fps_smooth:.1f}")
        sub = (f"1s avg mean:{mh} median:{self.smoothed['human']['median']} "
               f"mode:{self.smoothed['human']['mode']}")
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 56), (0, 0, 0), -1)
        cv2.putText(frame, header, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, sub, (10, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        self.last_counts = {
            "human": human_n, "non_human": non_human_n,
            "fps": round(self._fps_smooth, 1),
        }
        return frame, detections
