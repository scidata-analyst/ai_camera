from __future__ import annotations

import os
from dataclasses import dataclass

from .constants import (
    ALIGN_MODES,
    DEFAULT_AGE_ONNX,
    DEFAULT_AGE_RKNN,
    DEFAULT_FACE_ONNX,
    DEFAULT_FACE_RKNN,
    DEFAULT_HEADPOSE_ONNX,
    DEFAULT_HEADPOSE_RKNN,
    DEFAULT_YOLO_RKNN,
)


def _text(name: str, default: str) -> str:
    return os.getenv(name, default)


def _number(name: str, default: int | float) -> int | float:
    raw = os.getenv(name)
    if raw is None:
        return default
    return type(default)(raw)


def _int(name: str, default: int) -> int:
    return int(_number(name, default))


def _float(name: str, default: float) -> float:
    return float(_number(name, default))


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be 0/1/true/false, got {raw!r}")


@dataclass(frozen=True)
class Settings:
    model_name: str
    conf: float
    imgsz: int
    jpeg_quality: int
    stream_fps: float
    window_sec: float
    count_window: float
    face_enable: bool
    headpose_enable: bool
    face_min_size: int
    flip_h: bool
    flip_v: bool
    cv_threads: int
    yolo_backend: str
    rknn_model: str
    rknpu_core: str
    detect_every: int
    person_conf: float
    person_min_h: int
    person_require_face: bool
    det_stale_sec: float
    yolo_debug: bool
    face_every: int
    face_score: float
    face_backend: str
    face_onnx_path: str
    face_rknn_path: str
    age_backend: str
    age_onnx_path: str
    age_rknn_path: str
    face_blur: float
    age_align: str
    headpose_backend: str
    headpose_onnx_path: str
    headpose_rknn_path: str
    look_yaw: float
    look_pitch: float
    headpose_ttl: float
    camera_source: str
    frame_width: int
    frame_height: int

    @classmethod
    def from_env(cls) -> "Settings":
        align = _text("AGE_GENDER_ALIGN", "roll").lower()
        if align not in ALIGN_MODES:
            align = "roll"
        return cls(
            model_name=_text("MODEL_NAME", "models/yolo11n.pt"),
            conf=_float("CONF_THRESHOLD", 0.5),
            imgsz=_int("IMG_SIZE", 320),
            jpeg_quality=_int("JPEG_QUALITY", 60),
            stream_fps=_float("STREAM_FPS", 25.0),
            window_sec=_float("SMOOTH_WINDOW", 1.0),
            count_window=_float("COUNT_WINDOW", 1.0),
            face_enable=_bool("FACE_ENABLE", True),
            headpose_enable=_bool("HEADPOSE_ENABLE", True),
            face_min_size=_int("FACE_MIN_SIZE", 60),
            flip_h=_bool("FLIP_H", False),
            flip_v=_bool("FLIP_V", False),
            cv_threads=_int("CV_THREADS", 4),
            yolo_backend=_text("YOLO_BACKEND", "rknn").lower(),
            rknn_model=_text("RKNN_MODEL", DEFAULT_YOLO_RKNN),
            rknpu_core=_text("RKNPU_CORE", "AUTO"),
            detect_every=max(1, _int("DETECT_EVERY", 2)),
            person_conf=_float("PERSON_CONF", 0.65),
            person_min_h=_int("PERSON_MIN_H", 0),
            person_require_face=_bool("PERSON_REQUIRE_FACE", False),
            det_stale_sec=_float("DET_STALE_SEC", 0.2),
            yolo_debug=_bool("YOLO_DEBUG", False),
            face_every=max(1, _int("FACE_EVERY", 2)),
            face_score=_float("FACE_SCORE", 0.6),
            face_backend=_text("FACE_BACKEND", "rknn").lower(),
            face_onnx_path=_text("FACE_MODEL", DEFAULT_FACE_ONNX),
            face_rknn_path=_text("FACE_RKNN_MODEL", DEFAULT_FACE_RKNN),
            age_backend=_text("AGE_GENDER_BACKEND", "rknn").lower(),
            age_onnx_path=_text("AGE_GENDER_MODEL", DEFAULT_AGE_ONNX),
            age_rknn_path=_text("AGE_GENDER_RKNN_MODEL", DEFAULT_AGE_RKNN),
            face_blur=_float("FACE_BLUR", 80.0),
            age_align=align,
            headpose_backend=_text("HEADPOSE_BACKEND", "rknn").lower(),
            headpose_onnx_path=_text("HEADPOSE_MODEL", DEFAULT_HEADPOSE_ONNX),
            headpose_rknn_path=_text("HEADPOSE_RKNN_MODEL", DEFAULT_HEADPOSE_RKNN),
            look_yaw=_float("LOOK_YAW", 25.0),
            look_pitch=_float("LOOK_PITCH", 20.0),
            headpose_ttl=_float("HEADPOSE_TTL", 1.0),
            camera_source=_text("CAMERA_SOURCE", "0"),
            frame_width=_int("FRAME_WIDTH", 640),
            frame_height=_int("FRAME_HEIGHT", 480),
        )
