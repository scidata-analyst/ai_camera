from __future__ import annotations

import cv2
import numpy as np

from ...application.ports import FrameRenderer
from .text import draw as draw_text, measure as measure_text


class OpenCVFrameRenderer(FrameRenderer):
    LABEL_SIZE = 16
    HEADER_SIZE = 20
    SUB_SIZE = 16

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
            width, height = measure_text(label, self.LABEL_SIZE, 1)
            top = max(0, y1 - height - 8)
            cv2.rectangle(frame, (x1, top), (x1 + width + 4, y1), color, -1)
            draw_text(frame, (x1 + 2, top + 2), label, self.LABEL_SIZE,
                      (0, 0, 0), 1, (0, 0, 0))

    def draw_overlay(self, frame: np.ndarray, values: dict) -> None:
        header = (
            f"H:{values['human_n']}[{values['stable_human']}] "
            f"F:{values['face_n']}[{values['stable_face']}] "
            f"M:{values['male_n']}[{values['stable_male']}] "
            f"W:{values['female_n']}[{values['stable_female']}] "
            f"Look:{values['look_n']} FPS:{values['fps']:.1f}")
        seconds = f"{float(values['interval_sec']):g}s"
        sub = (f"{seconds} avg mean:{values['mean']} "
               f"median:{values['median']} mode:{values['mode']}")
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 56), (0, 0, 0), -1)
        draw_text(frame, (10, 6), header, self.HEADER_SIZE, (255, 255, 255), 1)
        draw_text(frame, (10, 31), sub, self.SUB_SIZE, (0, 255, 255))
