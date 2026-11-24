from __future__ import annotations

import math

import cv2
import numpy as np

from ...application.ports import FrameRenderer
from ...domain.geometry import iou


class OpenCVFrameRenderer(FrameRenderer):
    def draw_faces(self, frame: np.ndarray, faces: list[dict]) -> None:
        for face in faces:
            x1, y1, x2, y2 = face["box"]
            color = (0, 255, 0) if face.get("looking") else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            for px, py in face.get("landmarks", []):
                cv2.circle(frame, (int(px), int(py)), 2, (0, 0, 255), -1)

    def draw_humans(self, frame: np.ndarray, detections: list[dict],
                    faces: list[dict]) -> None:
        used: set[int] = set()
        for detection in detections:
            if detection["kind"] != "human":
                continue
            x1, y1, x2, y2 = detection["box"]
            face = self._face_for_human(detection, faces, used)
            if face is not None:
                used.add(id(face))
            gender = face.get("gender") if face is not None else None
            age = face.get("age") if face is not None else None
            if gender not in (None, "", "Unknown") and age:
                label = f"Human {gender} {age}y {detection['conf']:.2f}"
            else:
                label = f"Human {detection['conf']:.2f}"
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    @staticmethod
    def _distance(box, target) -> float:
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        tx = (target[0] + target[2]) / 2.0
        ty = (target[1] + target[3]) / 2.0
        return math.hypot(cx - tx, cy - ty)

    @staticmethod
    def _face_for_human(detection: dict, faces: list[dict],
                        used: set[int] | None = None):
        used = used if used is not None else set()
        box = detection["box"]
        owned = []
        for face in faces:
            if id(face) in used:
                continue
            human_box = face.get("human_box")
            if human_box and iou(human_box, box) >= 0.5:
                owned.append(face)
        if owned:
            return min(owned, key=lambda face: OpenCVFrameRenderer._distance(
                face["box"], box))
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        candidates = []
        for face in faces:
            if id(face) in used:
                continue
            fx1, fy1, fx2, fy2 = face["box"]
            fcx = (fx1 + fx2) / 2.0
            fcy = (fy1 + fy2) / 2.0
            if not (fx1 <= fcx <= fx2 and fy1 <= fcy <= fy2):
                continue
            if not (box[0] <= fcx <= box[2] and box[1] <= fcy <= box[3]):
                continue
            candidates.append((math.hypot(fcx - cx, fcy - cy), face))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    def draw_overlay(self, frame: np.ndarray, values: dict) -> None:
        header = (
            f"H:{values['human_n']}[{values['stable_human']}] "
            f"F:{values['face_n']}[{values['stable_face']}] "
            f"M:{values['male_n']}[{values['stable_male']}] "
            f"W:{values['female_n']}[{values['stable_female']}] "
            f"Look:{values['look_n']} FPS:{values['fps']:.1f}")
        sub = (f"1s avg mean:{values['mean']} "
               f"median:{values['median']} mode:{values['mode']}")
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 56), (0, 0, 0), -1)
        cv2.putText(frame, header, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, sub, (10, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
