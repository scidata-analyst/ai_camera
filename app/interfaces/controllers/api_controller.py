from __future__ import annotations

from fastapi import APIRouter

from ..flip_state import FlipState
from ..presenters.stats_presenter import StatsPresenter


class ApiController:
    def __init__(self, presenter: StatsPresenter, flip: FlipState):
        self.presenter = presenter
        self.flip = flip
        self.router = APIRouter()
        self._register()

    def _register(self) -> None:
        self.router.add_api_route("/api/health", self.health, methods=["GET"])
        self.router.add_api_route("/api/faces", self.faces, methods=["GET"])
        self.router.add_api_route("/api/stats", self.stats, methods=["GET"])
        self.router.add_api_route("/api/flip", self.get_flip, methods=["GET"])
        self.router.add_api_route("/api/flip/set", self.set_flip,
                                  methods=["POST"])
        self.router.add_api_route("/api/flip/toggle", self.toggle_flip,
                                  methods=["POST"])

    def health(self):
        return self.presenter.health()

    def faces(self):
        return self.presenter.faces()

    def stats(self):
        return self.presenter.stats()

    def get_flip(self):
        return self.flip.as_dict()

    def set_flip(self, h: int = None, v: int = None):
        return self.flip.set(h, v)

    def toggle_flip(self, axis: str = "h"):
        return self.flip.toggle(axis)
