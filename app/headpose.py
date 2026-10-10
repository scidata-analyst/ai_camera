"""WHENet head pose (yaw/pitch/roll) + looking-at-camera — RK3576 friendly.

Model: PINTO0309 whenet_1x3x224x224_prepost.onnx (~17MB, pre+post inside)
Input : face crop (BGR) -> 224x224 RGB raw 0-255 NCHW (norm model er ভিতরে)
Output: [yaw, roll, pitch] degree (-180..180)
looking = |yaw|<=LOOK_YAW and |pitch|<=LOOK_PITCH
"""
import os
import urllib.request

import cv2
import numpy as np

WHENET_URL = (
    "https://github.com/PINTO0309/HeadPoseEstimation-WHENet-yolov4-onnx-openvino/"
    "releases/download/v1.0.4/whenet_1x3x224x224_prepost.onnx"
)
DEFAULT_PATH = os.getenv("HEADPOSE_MODEL", "/app/models/whenet.onnx")
IN_SIZE = 224


def ensure_model(path: str = DEFAULT_PATH) -> str:
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"[headpose] downloading WHENet -> {path} ...")
    urllib.request.urlretrieve(WHENET_URL, path)
    return path


class HeadPoseEstimator:
    def __init__(self, model_path: str = DEFAULT_PATH):
        import onnxruntime as ort

        self.enabled = os.getenv("HEADPOSE_ENABLE", "1") == "1"
        self.yaw_thr = float(os.getenv("LOOK_YAW", "25"))
        self.pitch_thr = float(os.getenv("LOOK_PITCH", "20"))
        self.session = None
        self.input_name = None
        if not self.enabled:
            return
        try:
            path = ensure_model(model_path)
            self.session = ort.InferenceSession(
                path, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            print(f"[headpose] loaded {path}")
        except Exception as e:
            print(f"[headpose] disabled: {e}")
            self.session = None

    @property
    def loaded(self) -> bool:
        return self.session is not None

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        img = cv2.resize(crop, (IN_SIZE, IN_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
        return img.astype(np.float32)  # 0-255 raw, norm model er ভিতরে

    def predict_crop(self, crop: np.ndarray):
        """-> (yaw, pitch, roll) degree."""
        blob = self._preprocess(crop)
        out = self.session.run(None, {self.input_name: blob})[0][0]
        yaw, roll, pitch = float(out[0]), float(out[1]), float(out[2])
        return yaw, pitch, roll

    def predict_frame(self, frame: np.ndarray, faces, min_side: int = 40):
        """faces e yaw/pitch/roll/looking bosiye dey (in-place)."""
        if self.session is None:
            return faces
        H, W = frame.shape[:2]
        for fc in faces:
            x1, y1, x2, y2 = fc["box"]
            w, h = x2 - x1, y2 - y1
            if min(w, h) < min_side:
                fc.update({"yaw": 0.0, "pitch": 0.0, "roll": 0.0,
                           "looking": False})
                continue
            # margin soho crop (PINTO demo style)
            xa = max(int(x1 - w / 5), 0)
            xb = min(int(x2 + w / 5), W)
            ya = max(int(y1 - h / 10), 0)
            yb = min(int(y2 + h / 10), H)
            crop = frame[ya:yb, xa:xb]
            if crop.size == 0:
                fc.update({"yaw": 0.0, "pitch": 0.0, "roll": 0.0,
                           "looking": False})
                continue
            try:
                yaw, pitch, roll = self.predict_crop(crop)
                looking = bool(abs(yaw) <= self.yaw_thr
                               and abs(pitch) <= self.pitch_thr)
                fc.update({"yaw": round(yaw, 1), "pitch": round(pitch, 1),
                           "roll": round(roll, 1), "looking": looking})
            except Exception as e:
                print(f"[headpose] predict fail: {e}")
                fc.update({"yaw": 0.0, "pitch": 0.0, "roll": 0.0,
                           "looking": False})
        return faces
