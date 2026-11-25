from abc import ABC, abstractmethod

import numpy as np

from ..models import AttributeSample, Face


class AttributeEstimator(ABC):
    @abstractmethod
    def estimate(self, frame: np.ndarray, face: Face, mode: str) -> AttributeSample | None:
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> dict:
        raise NotImplementedError


class FaceAttributeRecognizer(ABC):
    @abstractmethod
    def recognize(self, frame: np.ndarray, faces: list[Face]) -> None:
        raise NotImplementedError

    @abstractmethod
    def fill(self, faces: list[Face]) -> list[Face]:
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> dict:
        raise NotImplementedError
