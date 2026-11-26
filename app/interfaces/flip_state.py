from __future__ import annotations

import cv2
import numpy as np


class FlipState:
    def __init__(self, horizontal: bool = False, vertical: bool = False):
        self.h = bool(horizontal)
        self.v = bool(vertical)

    def as_dict(self) -> dict:
        return {"h": self.h, "v": self.v}

    def set(self, h: int | None = None, v: int | None = None) -> dict:
        if h is not None:
            self.h = bool(int(h))
        if v is not None:
            self.v = bool(int(v))

        return self.as_dict()

    def toggle(self, axis: str = "h") -> dict:
        if axis in ("h", "horizontal"):
            self.h = not self.h
        elif axis in ("v", "vertical"):
            self.v = not self.v
        elif axis in ("both", "180"):
            self.h = not self.h
            self.v = not self.v

        return self.as_dict()

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if self.h and self.v:
            return cv2.flip(frame, -1)
        if self.h:
            return cv2.flip(frame, 1)
        if self.v:
            return cv2.flip(frame, 0)

        return frame
