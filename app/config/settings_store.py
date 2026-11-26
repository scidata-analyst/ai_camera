from __future__ import annotations

import json
import os
from pathlib import Path

from .constants import (
    ALIGN_MODES,
    DEFAULT_AGE_ONLY_RKNN,
    DEFAULT_FACE_ONNX,
    DEFAULT_FACE_RKNN,
    DEFAULT_HEADPOSE_ONNX,
    DEFAULT_MIVOLO_ONNX,
    DEFAULT_MIVOLO_RKNN,
    DEFAULT_HEADPOSE_RKNN,
    DEFAULT_YOLO_RKNN,
)

SETTINGS_FILE = Path(os.getenv(
    "SETTINGS_FILE", str(Path(__file__).with_name("saved_settings.json"))))

SPECS = {
    "CAMERA_SOURCE": {"group": "Camera / Stream", "type": "text", "default": "0", "live": False},
    "FRAME_WIDTH": {"group": "Camera / Stream", "type": "int", "default": "640", "live": False},
    "FRAME_HEIGHT": {"group": "Camera / Stream", "type": "int", "default": "480", "live": False},
    "STREAM_FPS": {"group": "Camera / Stream", "type": "float", "default": "25", "live": True},
    "JPEG_QUALITY": {"group": "Camera / Stream", "type": "int", "default": "60", "live": True},
    "FLIP_H": {"group": "Camera / Stream", "type": "bool", "default": "0", "live": True},
    "FLIP_V": {"group": "Camera / Stream", "type": "bool", "default": "0", "live": True},
    "MODEL_NAME": {"group": "Detection", "type": "text", "default": "models/yolo11n.pt", "live": False},
    "YOLO_BACKEND": {"group": "Detection", "type": "text", "default": "rknn", "live": False, "choices": ["rknn"]},
    "RKNN_MODEL": {"group": "Detection", "type": "text", "default": DEFAULT_YOLO_RKNN, "live": False},
    "RKNPU_CORE": {"group": "Detection", "type": "text", "default": "AUTO", "live": False},
    "CONF_THRESHOLD": {"group": "Detection", "type": "float", "default": "0.5", "live": True},
    "PERSON_CONF": {"group": "Detection", "type": "float", "default": "0.65", "live": True},
    "IMG_SIZE": {"group": "Detection", "type": "int", "default": "320", "live": False},
    "DETECT_EVERY": {"group": "Detection", "type": "int", "default": "2", "live": True},
    "DET_STALE_SEC": {"group": "Detection", "type": "float", "default": "0.2", "live": True},
    "PERSON_MIN_H": {"group": "Detection", "type": "int", "default": "0", "live": True},
    "PERSON_REQUIRE_FACE": {"group": "Detection", "type": "bool", "default": "0", "live": True},
    "YOLO_DEBUG": {"group": "Detection", "type": "bool", "default": "0", "live": True},
    "FACE_ENABLE": {"group": "Face", "type": "bool", "default": "1", "live": False},
    "FACE_BACKEND": {"group": "Face", "type": "text", "default": "rknn", "live": False, "choices": ["rknn", "off"]},
    "FACE_MODEL": {"group": "Face", "type": "text", "default": DEFAULT_FACE_ONNX, "live": False},
    "FACE_RKNN_MODEL": {"group": "Face", "type": "text", "default": DEFAULT_FACE_RKNN, "live": False},
    "FACE_SCORE": {"group": "Face", "type": "float", "default": "0.6", "live": True},
    "FACE_MIN_SIZE": {"group": "Face", "type": "int", "default": "60", "live": True},
    "FACE_BLUR": {"group": "Face", "type": "float", "default": "80", "live": True},
    "FACE_EVERY": {"group": "Face", "type": "int", "default": "1", "live": True},
    "AGE_GENDER_BACKEND": {"group": "Age / Gender", "type": "text", "default": "rknn", "live": False, "choices": ["rknn", "off"]},
    "AGE_GENDER_MODEL": {"group": "Age / Gender", "type": "text", "default": DEFAULT_MIVOLO_ONNX, "live": False},
    "AGE_GENDER_RKNN_MODEL": {"group": "Age / Gender", "type": "text", "default": DEFAULT_MIVOLO_RKNN, "live": False},
    "AGE_RKNN_MODEL": {"group": "Age / Gender", "type": "text", "default": DEFAULT_AGE_ONLY_RKNN, "live": False},
    "AGE_GENDER_ALIGN": {"group": "Age / Gender", "type": "text", "default": "arcface", "live": True, "choices": list(ALIGN_MODES)},
    "HEADPOSE_ENABLE": {"group": "Head Pose", "type": "bool", "default": "1", "live": False},
    "HEADPOSE_BACKEND": {"group": "Head Pose", "type": "text", "default": "rknn", "live": False, "choices": ["rknn", "off"]},
    "HEADPOSE_MODEL": {"group": "Head Pose", "type": "text", "default": DEFAULT_HEADPOSE_ONNX, "live": False},
    "HEADPOSE_RKNN_MODEL": {"group": "Head Pose", "type": "text", "default": DEFAULT_HEADPOSE_RKNN, "live": False},
    "LOOK_YAW": {"group": "Head Pose", "type": "float", "default": "25", "live": True},
    "LOOK_PITCH": {"group": "Head Pose", "type": "float", "default": "20", "live": True},
    "HEADPOSE_TTL": {"group": "Head Pose", "type": "float", "default": "1", "live": True},
    "INTERVAL": {"group": "System", "type": "float", "default": "1", "live": True},
    "CV_THREADS": {"group": "System", "type": "int", "default": "4", "live": False},
    "API_TOKEN": {"group": "System", "type": "text", "default": "", "live": True},
}

BASELINE = {key: os.getenv(key, spec["default"]) for key, spec in SPECS.items()}


def _coerce(key: str, value) -> str:
    spec = SPECS[key]
    kind = spec["type"]

    if isinstance(value, bool):
        raw = "1" if value else "0"
    else:
        raw = str(value).strip()

    if kind == "bool":
        lowered = raw.lower()

        if lowered in ("1", "true", "yes", "on"):
            return "1"
        if lowered in ("0", "false", "no", "off"):
            return "0"

        raise ValueError(f"{key} must be 0/1/true/false, got {raw!r}")

    if kind == "int":
        return str(int(raw))
    if kind == "float":
        return str(float(raw))

    choices = spec.get("choices")

    if choices and raw not in choices:
        raise ValueError(f"{key} must be one of {choices}, got {raw!r}")

    return raw


def current() -> dict[str, str]:
    return {key: os.getenv(key, spec["default"]) for key, spec in SPECS.items()}


def describe() -> list[dict]:
    values = current()

    return [{
        "key": key,
        "value": values[key],
        "group": spec["group"],
        "type": spec["type"],
        "live": spec["live"],
        "choices": spec.get("choices", []),
    } for key, spec in SPECS.items()]


def load_saved() -> None:
    try:
        data = json.loads(SETTINGS_FILE.read_text())
    except (OSError, ValueError):
        return

    if not isinstance(data, dict):
        return

    for key, value in data.items():
        if key in SPECS:
            try:
                os.environ[key] = _coerce(key, value)
            except (ValueError, TypeError):
                continue


def update(values: dict) -> list[dict]:
    unknown = [key for key in values if key not in SPECS]

    if unknown:
        raise ValueError(f"unknown setting: {unknown[0]}")

    merged = current()

    for key, value in values.items():
        merged[key] = _coerce(key, value)

    for key, value in merged.items():
        os.environ[key] = value

    try:
        SETTINGS_FILE.write_text(json.dumps(merged, indent=2) + "\n")
    except OSError:
        pass

    return describe()


def reset() -> list[dict]:
    for key, value in BASELINE.items():
        os.environ[key] = value

    try:
        SETTINGS_FILE.unlink()
    except OSError:
        pass

    return describe()
