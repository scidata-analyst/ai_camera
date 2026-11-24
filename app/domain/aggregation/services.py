from __future__ import annotations

import collections
import math
import statistics
import time

from ...config.constants import EMPTY_STATS


def _safe_mode(values):
    if not values:
        return 0
    try:
        return int(statistics.multimode(values)[0])
    except Exception:
        return int(collections.Counter(values).most_common(1)[0][0])


def _stats(values):
    if not values:
        return {"mean": 0.0, "median": 0, "mode": 0}
    return {
        "mean": round(float(statistics.mean(values)), 2),
        "median": int(statistics.median(values)),
        "mode": _safe_mode(values),
    }


class CountAggregator:
    def __init__(self, window_sec: float = 1.0):
        self.window_sec = float(window_sec)
        self.history: collections.deque = collections.deque()
        self._cached = {key: (dict(value) if isinstance(value, dict) else value)
                        for key, value in EMPTY_STATS.items()}
        self._cached["updated_at"] = time.time()
        self._last_publish = 0.0

    def add(self, human: int, non_human: int, face: int = 0,
            male: int = 0, female: int = 0):
        now = time.time()
        self.history.append((now, int(human), int(non_human),
                             int(face), int(male), int(female)))
        cutoff = now - self.window_sec
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

        if now - self._last_publish >= self.window_sec:
            self._last_publish = now
            human_values = [item[1] for item in self.history]
            other_values = [item[2] for item in self.history]
            face_values = [item[3] for item in self.history]
            male_values = [item[4] for item in self.history]
            female_values = [item[5] for item in self.history]
            human_stats = _stats(human_values)
            other_stats = _stats(other_values)
            face_stats = _stats(face_values)
            male_stats = _stats(male_values)
            female_stats = _stats(female_values)
            self._cached = {
                "human": human_stats,
                "non_human": other_stats,
                "face": face_stats,
                "male": male_stats,
                "female": female_stats,
                "stable_human": human_stats["mode"],
                "stable_non_human": other_stats["mode"],
                "stable_face": face_stats["mode"],
                "stable_male": male_stats["mode"],
                "stable_female": female_stats["mode"],
                "samples": len(human_values),
                "updated_at": now,
            }
        return self._cached

    @property
    def current(self):
        return self._cached


class AttributeAggregator:
    def __init__(self, window_sec: float = 1.0, iou_thr: float = 0.50,
                 max_tracks: int = 100):
        self.window_sec = float(window_sec)
        self.iou_thr = float(iou_thr)
        self.max_tracks = int(max_tracks)
        self.tracks: list = []

    @staticmethod
    def _iou(a, b) -> float:
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter <= 0:
            return 0.0
        area = ((a[2] - a[0]) * (a[3] - a[1]) +
                (b[2] - b[0]) * (b[3] - b[1]))
        return inter / max(area - inter, 1e-6)

    @staticmethod
    def _box(face) -> list:
        return [int(value) for value in face.get("human_box") or face["box"]]

    @staticmethod
    def _center(box) -> tuple[float, float]:
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    def _match(self, box: list, source=None):
        reference = source if source is not None else box
        cx, cy = self._center(reference)
        size = max(1, min(reference[2] - reference[0],
                          reference[3] - reference[1]))
        best, best_score = None, 0.0
        for track in self.tracks:
            overlap = self._iou(reference, track["box"])
            if overlap < self.iou_thr:
                continue
            tx, ty = self._center(track["box"])
            distance = math.hypot(cx - tx, cy - ty)
            if distance > size * 0.15 and overlap < 0.8:
                continue
            score = overlap - distance / size
            if score > best_score:
                best, best_score = track, score
        return best

    def _prune(self, now: float):
        cutoff = now - self.window_sec
        for track in self.tracks:
            while track["samples"] and track["samples"][0][0] < cutoff:
                track["samples"].popleft()
        self.tracks = [track for track in self.tracks if track["samples"]]
        self.tracks = self.tracks[-self.max_tracks:]

    def add(self, faces):
        now = time.time()
        for face in faces:
            gender = face.get("gender")
            age = face.get("age")
            if gender not in ("Male", "Female") or age in (None, 0):
                continue
            box = self._box(face)
            track = self._match(box, face.get("human_box"))
            if track is None:
                track = {"box": box, "samples": collections.deque()}
                self.tracks.append(track)
            track["box"] = box
            track["samples"].append((now, int(age), str(gender)))
        self._prune(now)

    def fill(self, faces):
        now = time.time()
        self._prune(now)
        for face in faces:
            track = self._match(self._box(face), face.get("human_box"))
            if track is None or not track["samples"]:
                continue
            samples = track["samples"]
            ages = [sample[1] for sample in samples]
            males = sum(1 for sample in samples if sample[2] == "Male")
            females = sum(1 for sample in samples if sample[2] == "Female")
            if males > females:
                gender = "Male"
            elif females > males:
                gender = "Female"
            else:
                gender = samples[-1][2]
            face.update({
                "gender": gender,
                "age": int(round(statistics.mean(ages))),
                "gconf": round(max(males, females) / len(samples), 3),
                "attr_samples": len(samples),
                "attr_method": "mean",
            })
        return faces
