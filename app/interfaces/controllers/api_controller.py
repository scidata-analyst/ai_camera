from __future__ import annotations

import cv2
from fastapi import APIRouter, Depends, HTTPException, Request

from ...config.settings import Settings
from ...config.settings_store import SPECS, describe, reset, update
from ..auth import require_token

router = APIRouter(dependencies=[Depends(require_token)])


def _apply_live(runtime, settings: Settings) -> None:
    runtime.settings           = settings
    runtime.presenter.settings = settings
    runtime.stream.settings    = settings

    human    = runtime.processor.human_detection
    detector = getattr(human, "detector", None)

    if detector is not None:

        if hasattr(detector, "conf"):
            detector.conf = settings.conf

        if hasattr(detector, "person_conf"):
            detector.person_conf = settings.person_conf

        if hasattr(detector, "debug"):
            detector.debug             = settings.yolo_debug
    human.conf                 = settings.conf
    human.min_height           = settings.person_min_h
    human.require_face_enabled = settings.person_require_face
    human.detect_every         = max(1, settings.detect_every)
    human.stale_sec            = settings.det_stale_sec

    counter = runtime.processor.count_aggregator

    if hasattr(counter, "set_interval"):
        counter.set_interval(settings.interval)
    else:
        counter.interval_sec = settings.interval

    face = runtime.processor.face_detection

    if face is not None:
        face.detect_every = max(1, settings.face_every)
        face_detector     = getattr(face, "detector", None)

        if face_detector is not None and hasattr(face_detector, "score_thr"):
            face_detector.score_thr = settings.face_score

    attributes = runtime.processor.attributes

    if attributes is not None:
        attributes.quality.blur_threshold = settings.face_blur
        attributes.min_size               = settings.face_min_size
        attributes.align_mode             = settings.age_align
        aggregator                        = attributes.aggregator

        if hasattr(aggregator, "set_interval"):
            aggregator.set_interval(settings.interval)
        else:
            aggregator.interval_sec = settings.interval

    headpose = runtime.processor.headpose

    if headpose is not None and headpose.analyzer is not None:
        headpose.analyzer.yaw_thr   = settings.look_yaw
        headpose.analyzer.pitch_thr = settings.look_pitch
        headpose.analyzer._ttl      = settings.headpose_ttl

    runtime.flip.set(int(settings.flip_h), int(settings.flip_v))

    try:
        cv2.setNumThreads(max(1, min(4, settings.cv_threads)))
    except Exception:
        pass



@router.get("/api/health")
def health(request: Request):

    return request.app.state.runtime.presenter.health()


@router.get("/api/faces")
def faces(request: Request):

    return request.app.state.runtime.presenter.faces()


@router.get("/api/stats")
def stats(request: Request):

    return request.app.state.runtime.presenter.stats()


@router.get("/api/settings")
def get_settings():

    return describe()


@router.post("/api/settings")
def post_settings(request: Request, payload: dict[str, object]):
    restart_required = [key for key in payload if key in SPECS and not SPECS[key]["live"]]

    try:
        describe_result = update(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    settings = Settings.from_env()
    _apply_live(request.app.state.runtime, settings)

    return {"ok": True, "settings": describe_result, "restart_required": restart_required}


@router.post("/api/settings/reset")
def reset_settings(request: Request):
    describe_result = reset()
    settings        = Settings.from_env()
    _apply_live(request.app.state.runtime, settings)
    restart_required = [key for key, spec in SPECS.items() if not spec["live"]]

    return {"ok": True, "settings": describe_result, "restart_required": restart_required}


@router.get("/api/flip")
def get_flip(request: Request):

    return request.app.state.runtime.flip.as_dict()


@router.post("/api/flip/set")
def set_flip(request: Request, h: int | None = None, v: int | None = None):

    return request.app.state.runtime.flip.set(h, v)


@router.post("/api/flip/toggle")
def toggle_flip(request: Request, axis: str = "h"):

    return request.app.state.runtime.flip.toggle(axis)
