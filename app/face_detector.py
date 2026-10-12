"""YuNet face detector (OpenCV DNN) — RK3576 friendly.

Output per face:
  box [x1,y1,x2,y2], conf, landmarks (5 points: eyes, nose, mouth corners)
Model: face_detection_yunet_2023mar.onnx (~300KB)
"""
import os
import urllib.request

import cv2
import numpy as np

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
DEFAULT_PATH = os.getenv("FACE_MODEL", "/app/models/yunet.onnx")


def ensure_model(path: str = DEFAULT_PATH) -> str:
    """Model na thakle download koro (dev + docker runtime)."""
    if os.path.exists(path) and os.path.getsize(path) > 100_000:
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"[face] downloading YuNet -> {path} ...")
    urllib.request.urlretrieve(YUNET_URL, path)
    return path


class YuNetFace:
    def __init__(self, model_path: str = DEFAULT_PATH,
                 score: float = 0.6, input_size=(320, 320)):
        self.score_thr = float(os.getenv("FACE_SCORE", str(score)))
        self.enabled = os.getenv("FACE_ENABLE", "1") == "1"
        self.detector = None
        # FACE_BACKEND=auto|cpu|opencl — auto = /dev/dri/renderD129 thakle OpenCL
        self.backend_want = os.getenv("FACE_BACKEND", "auto").lower()
        self.backend_used = "cpu"
        if not self.enabled:
            return
        try:
            path = ensure_model(model_path)
            backend, target = self._pick_backend()
            self.detector = cv2.FaceDetectorYN_create(
                path, "", input_size,
                self.score_thr, 0.3, 5000,
                backend, target,
            )
            self.model_path = path
            self.backend_used = "opencl" if target == cv2.dnn.DNN_TARGET_OPENCL else "cpu"
            print(f"[face] YuNet backend={self.backend_used} ({path})")
            self._last_hw = None  # LAG FIX: setInputSize cache
        except Exception as e:
            print(f"[face] YuNet disabled: {e}")
            self.detector = None

    @staticmethod
    def _pick_backend():
        """auto: renderD129/Mali thakle OpenCL try, fail hole CPU."""
        want = os.getenv("FACE_BACKEND", "auto").lower()
        if want == "cpu":
            return cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU
        if want == "opencl":
            return cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_OPENCL
        # auto
        try:
            from .accel import dri_devices
            import cv2 as _cv
            if dri_devices() and _cv.ocl.haveOpenCL():
                _cv.ocl.setUseOpenCL(True)
                if _cv.ocl.useOpenCL():
                    return _cv.dnn.DNN_BACKEND_OPENCV, _cv.dnn.DNN_TARGET_OPENCL
        except Exception:
            pass
        return cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU

    def detect(self, frame: np.ndarray):
        """Return list of {box, conf, landmarks}. Empty jodi disabled."""
        if self.detector is None:
            return []
        h, w = frame.shape[:2]
        # LAG FIX: size same thakle setInputSize skip (proti frame e costly)
        if self._last_hw != (w, h):
            self.detector.setInputSize((w, h))
            self._last_hw = (w, h)
        _, faces = self.detector.detect(frame)
        out = []
        if faces is None:
            return out
        for f in faces:
            x, y, ww, hh = map(int, f[0:4])
            conf = float(f[14])
            lms = []
            for i in range(5):
                lx, ly = int(f[4 + i * 2]), int(f[5 + i * 2])
                lms.append([lx, ly])
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x + ww, w), min(y + hh, h)
            out.append({"box": [x1, y1, x2, y2],
                        "conf": round(conf, 3), "landmarks": lms})
        return out

    @staticmethod
    def draw(frame: np.ndarray, faces) -> np.ndarray:
        for fc in faces:
            x1, y1, x2, y2 = fc["box"]
            conf = fc["conf"]
            # looking face = green box, na takale cyan
            color = (0, 255, 0) if fc.get("looking") else (255, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if fc.get("gender") and fc.get("gender") != "Unknown":
                label = f"{fc['gender'][0]}{fc['gconf']:.2f} {fc.get('age', 0)}y"
            else:
                label = f"Face {conf:.2f}"
            if fc.get("looking"):
                label += " [LOOK]"  # ASCII only (emoji OpenCV te ???? hoy)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), (255, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
            for (lx, ly) in fc["landmarks"]:
                cv2.circle(frame, (lx, ly), 2, (0, 0, 255), -1)
        return frame
