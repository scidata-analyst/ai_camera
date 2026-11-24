from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import cv2
from fastapi import FastAPI

from .application.attribute_recognition import AttributeRecognitionService
from .application.frame_processor import FrameProcessor
from .application.head_pose import HeadPoseService
from .config.settings import Settings
from .domain.aggregation.services import (
    AttributeAggregator,
    CountAggregator,
)
from .domain.age.services import AgeService
from .domain.detection.services import HumanDetectionService
from .domain.face.services import FaceDetectionService, FaceQualityService
from .domain.gender.services import GenderService
from .infrastructure.camera.opencv_camera import OpenCVCamera
from .infrastructure.rendering.opencv_renderer import OpenCVFrameRenderer
from .infrastructure.rknn.genderage_estimator import RknnGenderAgeEstimator
from .infrastructure.rknn.retinaface_detector import RknnFaceDetector
from .infrastructure.rknn.runtime import hw_summary
from .infrastructure.rknn.whenet_estimator import RknnHeadPoseEstimator
from .infrastructure.rknn.yolo_detector import RknnPersonDetector
from .interfaces.controllers import api_controller, stream_controller
from .interfaces.flip_state import FlipState
from .interfaces.presenters.stats_presenter import StatsPresenter


def _build_runtime() -> SimpleNamespace:
    settings = Settings.from_env()

    try:
        cv2.setNumThreads(max(1, min(4, settings.cv_threads)))
    except Exception:
        pass

    try:
        hardware = hw_summary()
        print(f"[hw] {hardware}")
    except Exception as exc:
        hardware = {}
        print(f"[hw] accel check skip: {exc}")

    camera = OpenCVCamera(settings.camera_source, settings.frame_width,
                          settings.frame_height)
    person_detector = RknnPersonDetector(settings)
    face_detector = RknnFaceDetector(settings) if settings.face_enable else None
    face_loaded = (face_detector is not None and
                   face_detector.describe().get("loaded", False))
    estimator = (RknnGenderAgeEstimator(settings) if face_loaded else None)
    age_service = AgeService(settings.age_scale, settings.age_bias)
    gender_service = GenderService()
    quality = FaceQualityService(settings.face_blur)
    attribute_aggregator = AttributeAggregator(window_sec=settings.window_sec)
    attributes = (AttributeRecognitionService(
        estimator, quality, age_service, gender_service, attribute_aggregator,
        settings.face_min_size, settings.age_align) if estimator is not None
        else None)
    headpose_adapter = (RknnHeadPoseEstimator(settings)
                        if settings.headpose_enable and face_loaded else None)
    headpose = (HeadPoseService(headpose_adapter, True)
                if headpose_adapter is not None else None)
    face_detection = (FaceDetectionService(
        face_detector, settings.max_faces, settings.face_every,
        attributes, headpose)
        if face_detector is not None and attributes is not None else None)
    human_detection = HumanDetectionService(
        person_detector, settings.conf, settings.person_min_h,
        settings.person_require_face, settings.detect_every,
        settings.det_stale_sec)
    count_aggregator = CountAggregator(settings.window_sec)
    renderer = OpenCVFrameRenderer()
    processor = FrameProcessor(human_detection, face_detection, attributes,
                               headpose, count_aggregator, renderer)
    flip = FlipState(settings.flip_h, settings.flip_v)
    presenter = StatsPresenter(settings, camera, processor, flip, hardware)
    stream = StreamController(camera, processor, settings, flip)

    return SimpleNamespace(
        settings=settings, hardware=hardware, camera=camera,
        processor=processor, presenter=presenter, stream=stream, flip=flip)


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime = _build_runtime()
    app.state.runtime = runtime
    runtime.camera.start()
    runtime.stream.start()
    try:
        yield
    finally:
        runtime.camera.stop()


app = FastAPI(title="AI Camera - Human + Face + Gender (RK3576)",
              lifespan=lifespan)
app.include_router(stream_controller.router)
app.include_router(api_controller.router)
