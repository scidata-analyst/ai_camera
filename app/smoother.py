"""1-sec window mean / median / mode smoother.

Proti frame er instant count (human, non_human) joma hoy,
proti 1 sec por por stable result publish hoy.
Flicker komate mode = stable, mean/median = reference.
"""
import collections
import statistics
import time


def _safe_mode(values):
    if not values:
        return 0
    try:
        # multimode sobcheye frequent gulo dey, prothom ta stable
        modes = statistics.multimode(values)
        return int(modes[0])
    except Exception:
        # fallback: most_common
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
        self.history: collections.deque = collections.deque()  # (t,h,n,f,m,w)
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
        self._last_publish = 0.0  # force first publish

    def add(self, human: int, non_human: int, face: int = 0,
            male: int = 0, female: int = 0):
        now = time.time()
        self.history.append((now, int(human), int(non_human),
                             int(face), int(male), int(female)))
        # purono sample bad dao
        cutoff = now - self.window_sec
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

        # 1 sec por por publish
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
                # stable = mode (sobcheye frequent), flicker-free
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
    """Mode for any hashable values (gender string, age int, ...)."""
    if not values:
        return None
    return collections.Counter(values).most_common(1)[0][0]


class FaceAttributeSmoother:
    """1-sec per-face age/gender mode smoother.

    Frame-e joto face (ar human) dekha jay, sob prediction 1 sec window-e
    joma kori. Box-IoU diye eki face-er track dhore, tar gender/age-er
    most-common (mode) value label-e dekhai — flicker komay.
    """

    def __init__(self, window_sec: float = 1.0, iou_thr: float = 0.35,
                 max_tracks: int = 100):
        self.window_sec = float(window_sec)
        self.iou_thr = float(iou_thr)
        self.max_tracks = int(max_tracks)
        self.tracks: list = []  # {box, samples: deque[(t,gender,age,conf)]}

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

    def _prune(self, now: float):
        cutoff = now - self.window_sec
        for track in self.tracks:
            while track["samples"] and track["samples"][0][0] < cutoff:
                track["samples"].popleft()
        self.tracks = [t for t in self.tracks if t["samples"]][-self.max_tracks:]

    def add(self, faces):
        """Store instant predictions. Only call when a fresh face infer ran."""
        now = time.time()
        for face in faces:
            gender = face.get("gender")
            age = face.get("age")
            if gender in (None, "", "Unknown") or age in (None, 0):
                continue
            box = [int(v) for v in face["box"]]
            track = self._match(box)
            if track is None:
                track = {"box": box, "samples": collections.deque()}
                self.tracks.append(track)
            track["box"] = box
            track["samples"].append(
                (now, str(gender), int(age), float(face.get("gconf", 0.0))))
        self._prune(now)

    def fill(self, faces):
        """Overwrite each face with the 1-sec mode of gender and age."""
        now = time.time()
        self._prune(now)
        for face in faces:
            track = self._match([int(v) for v in face["box"]])
            if track is None or not track["samples"]:
                continue
            samples = track["samples"]
            genders = [s[1] for s in samples]
            ages = [s[2] for s in samples]
            gender = _most_common(genders)
            age = _most_common(ages)
            confs = [s[3] for s in samples if s[1] == gender]
            face.update({
                "gender": gender,
                "age": int(age),
                "gconf": round(sum(confs) / max(len(confs), 1), 3),
                "age_mode_samples": collections.Counter(ages)[age],
                "gender_mode_samples": collections.Counter(genders)[gender],
                "attr_samples": len(samples),
            })
        return faces
