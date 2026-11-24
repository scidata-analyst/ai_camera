from abc import ABC, abstractmethod

import numpy as np


class FrameSource(ABC):
    @abstractmethod
    def read(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def connected(self) -> bool:
        raise NotImplementedError


class FrameRenderer(ABC):
    @abstractmethod
    def draw_faces(self, frame: np.ndarray, faces: list[dict]) -> None:
        raise NotImplementedError

    @abstractmethod
    def draw_humans(self, frame: np.ndarray, detections: list[dict]) -> None:
        raise NotImplementedError

    @abstractmethod
    def draw_overlay(self, frame: np.ndarray, values: dict) -> None:
        raise NotImplementedError
