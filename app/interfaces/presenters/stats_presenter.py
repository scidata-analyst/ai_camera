from __future__ import annotations

import time

from ...application.frame_processor import FrameProcessor
from ...application.ports import FrameSource
from ...config.settings import Settings
from ..flip_state import FlipState


class StatsPresenter:
    def __init__(self, settings: Settings, camera: FrameSource,
                 processor: FrameProcessor, flip: FlipState,
                 hardware: dict):
        self.settings = settings
        self.camera = camera
        self.processor = processor
        self.flip = flip
        self.hardware = hardware

    def health(self) -> dict:
        human = self.processor.human_detection.describe()
        face = (self.processor.face_detection.describe()
                if self.processor.face_detection is not None else {})
        attributes = (self.processor.attributes.describe()
                      if self.processor.attributes is not None else {})
        headpose = (self.processor.headpose.describe()
                    if self.processor.headpose is not None else {})
        backends = {
            "object_detection": human.get("backend", "cpu"),
            "face_detection": face.get("backend", "off"),
            "age_gender": attributes.get("backend", "off"),
            "head_pose": headpose.get("backend", "off"),
        }
        return {
            "status": "ok",
            "camera_connected": self.camera.connected,
            "model": self.settings.model_name,
            "conf": self.settings.conf,
            "imgsz": self.settings.imgsz,
            "backend": human.get("backend", "cpu"),
            "rknn_model": human.get("rknn_model"),
            "npu_active": human.get("npu_active", False),
            "model_backends": backends,
            "all_models_npu_active": all(
                backend == "rknn" for backend in backends.values()),
            "hw": self.hardware,
            "face_backend": face.get("backend", "off"),
            "face_rknn_model": face.get("rknn_model"),
            "headpose_backend": headpose.get("backend", "off"),
            "headpose_rknn_model": headpose.get("rknn_model"),
            "headpose_providers": headpose.get("providers", []),
            "face_enabled": self.settings.face_enable,
            "face_loaded": (self.processor.face_detection.loaded
                            if self.processor.face_detection is not None
                            else False),
            "face_min_size": self.settings.face_min_size,
            "age_gender_enabled": self.processor.attributes is not None,
            "age_gender_loaded": attributes.get("loaded", False),
            "age_gender_backend": attributes.get("backend", "off"),
            "age_gender_model": attributes.get("onnx_model"),
            "age_gender_rknn_model": attributes.get("rknn_model"),
            "age_gender_align": attributes.get("align"),
            "face_blur_min": attributes.get("blur"),
            "age_calibration": {
                "scale": attributes.get("scale"),
                "bias": attributes.get("bias"),
            },
            "headpose_enabled": self.settings.headpose_enable,
            "headpose_loaded": headpose.get("loaded", False),
            "infer_fps": round(self.processor.infer_fps, 1),
        }

    def stats(self) -> dict:
        attributes = (self.processor.attributes.describe()
                      if self.processor.attributes is not None else {})
        return {
            "camera_connected": self.camera.connected,
            "instant": self.processor.last_counts,
            "smoothed": self.processor.smoothed,
            "flip": self.flip.as_dict(),
            "infer_fps": round(self.processor.infer_fps, 1),
            "age_gender": {
                "runs": attributes.get("runs", 0),
                "last_ms": attributes.get("last_ms", 0.0),
            },
            "window_sec": self.settings.window_sec,
            "time": time.time(),
        }

    def faces(self) -> dict:
        return {
            "faces": self.processor.last_faces,
            "count": self.processor.last_counts.get("face", 0),
            "time": time.time(),
        }
