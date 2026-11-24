import collections
import statistics
import time


def _safe_mode(values):
    if not values:
        return 0
    try:
        modes = statistics.multimode(values)
        return int(modes[0])
    except Exception:
        c = collections.Counter(values)
        return int(c.most_common(1)[0][0])


def _stats(values):
    if not values:
        return {"mean": 0.0, "median": 0, "mode": 0}
    return {
        "mean": round(float(statistics.mean(values)), 2),
        "median": int(statistics.median(values)),
        "mode": _safe_mode(values),
    }


class CountSmoother:
    def __init__(self, window_sec: float = 1.0):
        self.window_sec = float(window_sec)
        self.history: collections.deque = collections.deque()
        self._cached = {
            "human": {"mean": 0.0, "median": 0, "mode": 0},
            "non_human": {"mean": 0.0, "median": 0, "mode": 0},
            "face": {"mean": 0.0, "median": 0, "mode": 0},
            "male": {"mean": 0.0, "median": 0, "mode": 0},
            "female": {"mean": 0.0, "median": 0, "mode": 0},
            "stable_human": 0,
            "stable_non_human": 0,
            "stable_face": 0,
            "stable_male": 0,
            "stable_female": 0,
            "samples": 0,
            "updated_at": time.time(),
        }
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
            hs = [x[1] for x in self.history]
            ns = [x[2] for x in self.history]
            fs = [x[3] if len(x) > 3 else 0 for x in self.history]
            ms = [x[4] if len(x) > 4 else 0 for x in self.history]
            ws = [x[5] if len(x) > 5 else 0 for x in self.history]
            h_stat = _stats(hs)
            n_stat = _stats(ns)
            f_stat = _stats(fs)
            m_stat = _stats(ms)
            w_stat = _stats(ws)
            self._cached = {
                "human": h_stat,
                "non_human": n_stat,
                "face": f_stat,
                "male": m_stat,
                "female": w_stat,
                "stable_human": h_stat["mode"],
                "stable_non_human": n_stat["mode"],
                "stable_face": f_stat["mode"],
                "stable_male": m_stat["mode"],
                "stable_female": w_stat["mode"],
                "samples": len(hs),
                "updated_at": now,
            }
        return self._cached

    @property
    def current(self):
        return self._cached


def _most_common(values):
    if not values:
        return None
    return collections.Counter(values).most_common(1)[0][0]


class FaceAttributeSmoother:

    def __init__(self, window_sec: float = 1.0, iou_thr: float = 0.50,
                 max_tracks: int = 100, hold_sec: float = 5.0,
                 review_n: int = 5, age_tol: float = 6.0):
        self.window_sec = float(window_sec)
        self.iou_thr = float(iou_thr)
        self.max_tracks = int(max_tracks)
        self.hold_sec = max(float(hold_sec), self.window_sec)
        self.review_n = max(2, int(review_n))
        self.age_tol = float(age_tol)
        self.review_min_conf = 0.55
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

    def _match(self, box: list):
        best, best_iou = None, 0.0
        for track in self.tracks:
            iou = self._iou(box, track["box"])
            if iou > best_iou:
                best, best_iou = track, iou
        return best if best_iou >= self.iou_thr else None

    @staticmethod
    def _signed(gender: str, conf: float) -> float:
        return float(conf) if gender == "Male" else -float(conf)

    def _new_track(self, box, now, good=None) -> dict:
        return {"box": box, "created": now, "samples": collections.deque(),
                "last_good": good, "locked": None, "last_seen": now,
                "review": collections.deque(maxlen=self.review_n)}

    @staticmethod
    def _average(samples, now: float) -> dict:
        signed_mean = sum(s[1] for s in samples) / len(samples)
        age_mean = sum(s[2] for s in samples) / len(samples)
        return {
            "t": now,
            "gender": "Male" if signed_mean >= 0 else "Female",
            "age": int(round(age_mean)),
            "conf": abs(signed_mean),
        }

    def _prune(self, now: float):
        cutoff = now - self.window_sec
        stale = now - self.hold_sec
        for track in self.tracks:
            while track["samples"] and track["samples"][0][0] < cutoff:
                track["samples"].popleft()
        self.tracks = [t for t in self.tracks
                       if t.get("last_seen", t["created"]) >= stale]
        self.tracks.sort(key=lambda t: (
            (t["locked"] or t["last_good"] or {"t": 0.0})["t"]))
        self.tracks = self.tracks[-self.max_tracks:]

    def _review_lock(self, track, gender: str, conf: float, age: int,
                     now: float):
        track["review"].append(
            (now, self._signed(gender, conf), int(age)))
        if len(track["review"]) < self.review_n:
            return
        locked = track["locked"]
        if locked is None:
            return
        vals = list(track["review"])
        signed_mean = sum(v[1] for v in vals) / len(vals)
        age_mean = sum(v[2] for v in vals) / len(vals)
        new_gender = "Male" if signed_mean >= 0 else "Female"
        gender_flip = new_gender != locked["gender"]
        age_shift = abs(age_mean - locked["age"]) > self.age_tol
        if gender_flip and abs(signed_mean) < self.review_min_conf:
            return
        if not gender_flip and not age_shift:
            return
        track["locked"] = None
        track["last_good"] = None
        track["created"] = now - self.window_sec
        track["samples"] = collections.deque(vals)
        track["review"].clear()

    def add(self, faces):
        now = time.time()
        for face in faces:
            gender = face.get("gender")
            age = face.get("age")
            if gender in (None, "", "Unknown") or age in (None, 0):
                continue
            conf = max(0.0, min(1.0, float(face.get("gconf", 0.0))))
            box = [int(v) for v in face["box"]]
            track = self._match(box)
            if track is None:
                track = self._new_track(box, now)
                self.tracks.append(track)
            track["box"] = box
            track["last_seen"] = now
            if track["locked"] is not None:
                self._review_lock(track, str(gender), conf, int(age), now)
                continue
            track["samples"].append(
                (now, self._signed(str(gender), conf), int(age)))
            track["last_good"] = {
                "t": now, "gender": str(gender), "age": int(age), "conf": conf}
        self._prune(now)

    def fill(self, faces):
        now = time.time()
        self._prune(now)
        for face in faces:
            box = [int(v) for v in face["box"]]
            track = self._match(box)
            if track is None:
                good = None
                if face.get("gender") not in (None, "", "Unknown") \
                        and face.get("age") not in (None, 0):
                    good = {
                        "t": now, "gender": str(face["gender"]),
                        "age": int(face["age"]),
                        "conf": float(face.get("gconf", 0.0))}
                track = self._new_track(box, now, good)
                self.tracks.append(track)
            track["last_seen"] = now

            if track["locked"] is not None:
                value, method, samples_n = track["locked"], "locked", 0
            elif track["samples"]:
                samples_n = len(track["samples"])
                value = self._average(track["samples"], now)
                if now - track["created"] >= self.window_sec:
                    track["locked"] = value
                    track["last_good"] = value
                    track["samples"].clear()
                    track["review"].clear()
                    method = "locked"
                else:
                    method = "mean"
            elif track["last_good"]:
                value, method, samples_n = track["last_good"], "sticky", 0
            else:
                value = {
                    "gender": face.get("gender") or "Unknown",
                    "age": int(face.get("age") or 0),
                    "conf": float(face.get("gconf") or 0.0),
                }
                method, samples_n = "sticky-self", 0

            face.update({
                "gender": value["gender"],
                "age": int(value["age"]),
                "gconf": round(float(value["conf"]), 3),
                "attr_samples": samples_n,
                "attr_method": method,
                "attr_sticky": True,
            })
        return faces
