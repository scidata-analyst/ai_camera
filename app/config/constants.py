import numpy as np

PERSON_CLASS_NAME = "person"

DEFAULT_YOLO_RKNN     = "/app/models/yolo11n-rk3576.rknn"
DEFAULT_FACE_RKNN     = "/app/models/RetinaFace_mobile320-rk3576.rknn"
DEFAULT_FACE_ONNX     = "/app/models/RetinaFace_mobile320.onnx"
DEFAULT_AGE_ONLY_RKNN = "/app/models/dlib-age-rk3576.rknn"
DEFAULT_MIVOLO_RKNN   = "/app/models/mivolo-face-rk3576.rknn"
DEFAULT_MIVOLO_ONNX   = "/app/models/mivolo-face.onnx"
DEFAULT_HEADPOSE_RKNN = "/app/models/whenet-rk3576.rknn"
DEFAULT_HEADPOSE_ONNX = "/app/models/whenet.onnx"

AGE_INPUT_SIZE        = 224
LEGACY_AGE_INPUT_SIZE = 96
FACE_DETECTOR_SIZE    = 320
WHENET_INPUT_SIZE     = 224

FAIRFACE_MEAN     = (0.485, 0.456, 0.406)
FAIRFACE_STD      = (0.229, 0.224, 0.225)
FAIRFACE_AGE_BINS = np.array(
    [1.0, 6.0, 14.5, 24.5, 34.5, 44.5, 54.5, 64.5, 70.0], dtype=np.float32)

AGE_ONLY_INPUT_SIZE = 64

DLIB_AGE_MEAN    = (122.781998, 117.000999, 104.297997)
DLIB_AGE_SCALE   = 256.0
DLIB_AGE_WEIGHTS = np.array([0.25] + list(range(1, 81)), dtype=np.float32)

MIVOLO_AGE_MIN = 1.0
MIVOLO_AGE_MAX = 95.0
MIVOLO_AGE_AVG = 48.0

ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32) / 112.0

RENDER_GLOB     = "/dev/dri/renderD*"
RKNPU_GLOB      = "/dev/rknpu*"
EXPLICIT_RENDER = "/dev/dri/renderD129"

ALIGN_MODES = ("box", "roll", "arcface")
EMPTY_STATS = {
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
    "updated_at": 0.0,
}
