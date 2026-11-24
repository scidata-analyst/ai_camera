import numpy as np

YOLO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)
PERSON_CLASS_ID = 0
PERSON_CLASS_NAME = "person"

DEFAULT_YOLO_RKNN = "/app/models/yolo11n-rk3576.rknn"
DEFAULT_FACE_RKNN = "/app/models/RetinaFace_mobile320-rk3576.rknn"
DEFAULT_FACE_ONNX = "/app/models/RetinaFace_mobile320.onnx"
DEFAULT_AGE_RKNN = "/app/models/fairface-rk3576.rknn"
DEFAULT_AGE_ONNX = "/app/models/fairface.onnx"
DEFAULT_HEADPOSE_RKNN = "/app/models/whenet-rk3576.rknn"
DEFAULT_HEADPOSE_ONNX = "/app/models/whenet.onnx"

AGE_INPUT_SIZE = 224
LEGACY_AGE_INPUT_SIZE = 96
FACE_DETECTOR_SIZE = 320
WHENET_INPUT_SIZE = 224
FAIRFACE_MEAN = (0.485, 0.456, 0.406)
FAIRFACE_STD = (0.229, 0.224, 0.225)
FAIRFACE_AGE_BINS = np.array(
    [1.0, 6.0, 14.5, 24.5, 34.5, 44.5, 54.5, 64.5, 70.0], dtype=np.float32)

ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32) / 112.0

RENDER_GLOB = "/dev/dri/renderD*"
RKNPU_GLOB = "/dev/rknpu*"
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
