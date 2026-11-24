from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from ..auth import require_token

router = APIRouter(dependencies=[Depends(require_token)])


@router.get("/api/health")
def health(request: Request):
    return request.app.state.runtime.presenter.health()


@router.get("/api/faces")
def faces(request: Request):
    return request.app.state.runtime.presenter.faces()


@router.get("/api/stats")
def stats(request: Request):
    return request.app.state.runtime.presenter.stats()


@router.get("/api/flip")
def get_flip(request: Request):
    return request.app.state.runtime.flip.as_dict()


@router.post("/api/flip/set")
def set_flip(request: Request, h: int | None = None,
             v: int | None = None):
    return request.app.state.runtime.flip.set(h, v)


@router.post("/api/flip/toggle")
def toggle_flip(request: Request, axis: str = "h"):
    return request.app.state.runtime.flip.toggle(axis)
