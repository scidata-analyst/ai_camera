from abc import ABC, abstractmethod

import numpy as np

from ..models import Detection


class PersonDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> dict:
        raise NotImplementedError
