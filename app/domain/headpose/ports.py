from abc import ABC, abstractmethod

import numpy as np

from ..models import Face


class HeadPoseAnalyzer(ABC):
    @abstractmethod
    def predict_frame(self, frame: np.ndarray, faces: list[Face]) -> list[Face]:
        raise NotImplementedError

    @abstractmethod
    def remember(self, faces: list[Face]) -> list[Face]:
        raise NotImplementedError

    @abstractmethod
    def fill_from_cache(self, faces: list[Face]) -> list[Face]:
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> dict:
        raise NotImplementedError

    def submit(self, frame: np.ndarray, faces: list[Face]) -> None:
        return None
