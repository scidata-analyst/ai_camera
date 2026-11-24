from __future__ import annotations

import collections
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
    def __init__(self, interval_sec: float = 1.0):
        self.interval_sec = float(interval_sec)
        self.history: list = []
        self._cached = {key: (dict(value) if isinstance(value, dict) else value)
                        for key, value in EMPTY_STATS.items()}
        self._cached["updated_at"] = time.time()
        self._last_publish = time.time()

    def set_interval(self, interval_sec: float) -> None:
        self.interval_sec = float(interval_sec)
        self.history.clear()
        self._last_publish = time.time()

    def add(self, human: int, non_human: int, face: int = 0,
            male: int = 0, female: int = 0):
        now = time.time()
        self.history.append((now, int(human), int(non_human),
                             int(face), int(male), int(female)))
        if now - self._last_publish < self.interval_sec:
            return self._cached
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
        self.history.clear()
        self._last_publish = now
        return self._cached

    @property
    def current(self):
        return self._cached


class AttributeAggregator:
    def __init__(self, interval_sec: float = 1.0):
        self.interval_sec = float(interval_sec)
        self.tracks: dict = {}
        self._last_good: collections.OrderedDict = collections.OrderedDict()
        self._last_publish = time.time()

    def set_interval(self, interval_sec: float) -> None:
        self.interval_sec = float(interval_sec)
        self._last_publish = time.time()
        self.tracks.clear()

    def add(self, faces):
        now = time.time()
        for face in faces:
            person_id = face.get("person_id")
            if person_id is None:
                continue
            gender = face.get("gender")
            age = face.get("age")
            if gender not in ("Male", "Female") or age in (None, 0):
                continue
            samples = self.tracks.setdefault(person_id, collections.deque())
            samples.append((now, int(age), str(gender)))

    def results(self) -> dict:
        now = time.time()
        if now - self._last_publish < self.interval_sec:
            return {person_id: dict(result)
                    for person_id, result in self._last_good.items()}
        fresh = {}
        for person_id, samples in self.tracks.items():
            ages = [sample[1] for sample in samples]
            males = sum(1 for sample in samples if sample[2] == "Male")
            females = sum(1 for sample in samples if sample[2] == "Female")
            if males > females:
                gender = "Male"
            elif females > males:
                gender = "Female"
            else:
                gender = samples[-1][2]
            fresh[person_id] = {
                "gender": gender,
                "age": int(round(statistics.mean(ages))),
                "gconf": round(max(males, females) / len(samples), 3),
                "attr_samples": len(samples),
                "attr_method": "mean",
            }
        self.tracks.clear()
        for person_id, result in fresh.items():
            if person_id in self._last_good:
                self._last_good.move_to_end(person_id)
            self._last_good[person_id] = dict(result)
        self._last_publish = now
        return {person_id: dict(result)
                for person_id, result in self._last_good.items()}

    def fill(self, faces):
        results = self.results()
        for face in faces:
            result = results.get(face.get("person_id"))
            if result is not None:
                face.update(result)
        return faces
