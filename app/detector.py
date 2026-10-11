"""YOLOv11 based Human vs Non-Human detector."""
import threading
import time
from typing import List, Dict

import cv2
import numpy as np

from .smoother import CountSmoother
from .face_detector import YuNetFace
from .mivolo import MiVOLOAgeGender
from .headpose import HeadPoseEstimator


class HumanDetector:
    """YOLO Human + YuNet Face + MiVOLO age/gender + WHENet looking."""

    def __init__(self, model_name: str = "yolo11n.pt", conf: float = 0.5, imgsz: int = 640,
                 window_sec: float = 1.0, face_enable: bool = True,
                 mivolo_enable: bool = True, headpose_enable: bool = True,
                 face_min_size: int = 60):
        from ultralytics import YOLO

        self.conf = conf
        self.imgsz = imgsz
        self.model_name = model_name
        self.face_min_size = int(face_min_size)
        # lazy download on first run; Docker build e cache kora jabe
        self.model = YOLO(model_name)
        self.names = self.model.names  # {0: 'person', ...}
        self._lock = threading.Lock()
        self.last_counts = {"human": 0, "non_human": 0, "face": 0,
                            "male": 0, "female": 0, "fps": 0.0}
        self.last_faces = []
        self.face = YuNetFace() if face_enable else None
        self.face_enabled = self.face is not None and self.face.detector is not None
        self.mivolo = MiVOLOAgeGender() if (mivolo_enable and self.face_enabled) else None
        self.mivolo_enabled = self.mivolo is not None and self.mivolo.loaded
        self.headpose = HeadPoseEstimator() if (headpose_enable and self.face_enabled) else None
        self.headpose_enabled = self.headpose is not None and self.headpose.loaded
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

        # YuNet face -> WHENet looking -> MiVOLO age/gender (body = YOLO box).
        # looking age ber hoy, jate blurry/side frame e age refresh off thake.
        faces = []
        if self.face is not None:
            try:
                faces = self.face.detect(frame)
                if self.headpose is not None and faces:
                    faces = self.headpose.predict_frame(frame, faces)
                # Gender+Age: sudhu boro/clear/frontal face e refresh
                if self.mivolo is not None and faces:
                    bodies = [d["box"] for d in detections if d["kind"] == "human"]
                    faces = self.mivolo.predict_frame(
                        frame, faces, bodies, min_side=self.face_min_size)
                YuNetFace.draw(frame, faces)
            except Exception as e:
                print(f"[face] infer fail: {e}")
                faces = []
        face_n = len(faces)
        male_n = sum(1 for f in faces if f.get("gender") == "Male")
        female_n = sum(1 for f in faces if f.get("gender") == "Female")
        look_n = sum(1 for f in faces if f.get("looking"))
        self.last_faces = faces

        # FPS + 1-sec smoothed header (mean/median/mode)
        now = time.time()
        inst = 1.0 / max(now - self._t_prev, 1e-6)
        self._t_prev = now
        self._fps_smooth = self._fps_smooth * 0.9 + inst * 0.1

        self.smoothed = self.smoother.add(human_n, non_human_n, face_n,
                                            male_n, female_n)
        sh = self.smoothed["stable_human"]
        sn = self.smoothed["stable_non_human"]
        sf = self.smoothed.get("stable_face", face_n)
        sm = self.smoothed.get("stable_male", male_n)
        sw = self.smoothed.get("stable_female", female_n)
        mh = self.smoothed["human"]["mean"]

        header = (f"H:{human_n}[{sh}] F:{face_n}[{sf}] "
                  f"M:{male_n}[{sm}] W:{female_n}[{sw}] Look:{look_n} FPS:{self._fps_smooth:.1f}")
        sub = (f"1s avg mean:{mh} median:{self.smoothed['human']['median']} "
               f"mode:{self.smoothed['human']['mode']}")
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 56), (0, 0, 0), -1)
        cv2.putText(frame, header, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, sub, (10, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        self.last_counts = {
            "human": human_n, "non_human": non_human_n, "face": face_n,
            "male": male_n, "female": female_n, "looking": look_n,
            "fps": round(self._fps_smooth, 1),
        }
        return frame, detections
