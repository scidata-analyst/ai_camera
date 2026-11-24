from abc import ABC, abstractmethod

import numpy as np

from ..models import Face


class FaceDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Face]:
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> dict:
        raise NotImplementedError
