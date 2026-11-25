from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = (Path(__file__).resolve().parents[2] /
             "interfaces" / "controllers" / "UbuntuSans.ttf")
_FONTS: dict[int, ImageFont.FreeTypeFont] = {}


def font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _FONTS:

        _FONTS[size] = ImageFont.truetype(str(FONT_PATH), size)
    return _FONTS[size]


def measure(text: str, size: int, stroke: int = 0) -> tuple[int, int]:
    box = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox(
        (0, 0), text, font=font(size), stroke_width=stroke)
    return box[2] - box[0], box[3] - box[1]


def draw(frame: np.ndarray, xy: tuple[int, int], text: str, size: int,
         fill: tuple[int, int, int], stroke: int = 0,
         stroke_fill: tuple[int, int, int] = (0, 0, 0)) -> None:
    image = Image.fromarray(frame)
    ImageDraw.Draw(image).text(
        xy, text, font=font(size), fill=fill,
        stroke_width=stroke, stroke_fill=stroke_fill)
    frame[:] = np.asarray(image)
