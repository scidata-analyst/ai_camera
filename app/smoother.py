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
        self.history: collections.deque = collections.deque()  # (t, h, n)
        self._cached = {
            "human": {"mean": 0.0, "median": 0, "mode": 0},
            "non_human": {"mean": 0.0, "median": 0, "mode": 0},
            "stable_human": 0,
            "stable_non_human": 0,
            "samples": 0,
            "updated_at": time.time(),
        }
        self._last_publish = 0.0  # force first publish

    def add(self, human: int, non_human: int):
        now = time.time()
        self.history.append((now, int(human), int(non_human)))
        # purono sample bad dao
        cutoff = now - self.window_sec
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

        # 1 sec por por publish
        if now - self._last_publish >= self.window_sec:
            self._last_publish = now
            hs = [h for _, h, _ in self.history]
            ns = [n for _, _, n in self.history]
            h_stat = _stats(hs)
            n_stat = _stats(ns)
            self._cached = {
                "human": h_stat,
                "non_human": n_stat,
                # stable = mode (sobcheye frequent), flicker-free
                "stable_human": h_stat["mode"],
                "stable_non_human": n_stat["mode"],
                "samples": len(hs),
                "updated_at": now,
            }
        return self._cached

    @property
    def current(self):
        return self._cached
