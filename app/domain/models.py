from __future__ import annotations

from dataclasses import dataclass
from typing import Any

Detection = dict[str, Any]
Face      = dict[str, Any]


@dataclass(frozen=True)
class AttributeSample:
    raw_age: float
    gender_logits: tuple[float, float]
    align: str
