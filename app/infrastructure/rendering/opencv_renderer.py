from __future__ import annotations

import cv2
import numpy as np

from ...application.ports import FrameRenderer


class OpenCVFrameRenderer(FrameRenderer):
    def draw_faces(self, frame: np.ndarray, faces: list[dict]) -> None:
        for face in faces:
            x1, y1, x2, y2 = face["box"]
            color = (0, 255, 0) if face.get("looking") else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            for px, py in face.get("landmarks", []):
                cv2.circle(frame, (int(px), int(py)), 2, (0, 0, 255), -1)

    def draw_humans(self, frame: np.ndarray, detections: list[dict]) -> None:
        for detection in detections:
            if detection["kind"] != "human":
                continue
            x1, y1, x2, y2 = detection["box"]
            gender = detection.get("gender")
            age = detection.get("age")
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
