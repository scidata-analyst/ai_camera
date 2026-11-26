from __future__ import annotations

import math
from itertools import product

import cv2
import numpy as np

from ..config.constants import ARCFACE_DST, FACE_DETECTOR_SIZE


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)

    if inter <= 0:

        return 0.0
    area = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]))

    return inter / max(area - inter, 1e-6)


def letterbox(img: np.ndarray, size: int):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top     = (size - nh) // 2
    left    = (size - nw) // 2
    out     = np.full((size, size, 3), 114, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = resized

    return out, r, left, top


def face_letterbox(frame: np.ndarray):
    h, w = frame.shape[:2]
    ratio = min(FACE_DETECTOR_SIZE / w, FACE_DETECTOR_SIZE / h)
    nw, nh = int(w * ratio), int(h * ratio)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    pad_x, pad_y = (FACE_DETECTOR_SIZE - nw) // 2, (FACE_DETECTOR_SIZE - nh) // 2
    canvas = np.full((FACE_DETECTOR_SIZE, FACE_DETECTOR_SIZE, 3), 114, dtype = np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

    return canvas, ratio, pad_x, pad_y


def sharpness(frame: np.ndarray, box) -> float:
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    xa, ya, xb, yb = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

    if xb - xa < 2 or yb - ya < 2:

        return 0.0
    gray = cv2.cvtColor(frame[ya:yb, xa:xb], cv2.COLOR_BGR2GRAY)

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def landmarks(face: dict) -> np.ndarray | None:
    value = face.get("landmarks")

    if value is None or len(value) != 5:

        return None

    try:
        points = np.asarray(value, dtype=np.float32).reshape(5, 2)
    except (TypeError, ValueError):

        return None


    return points if np.all(np.isfinite(points)) else None


def crop_face(frame: np.ndarray, face: dict, mode: str, size: int) -> np.ndarray:
    x1, y1, x2, y2 = map(float, face["box"])
    width, height = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    scale  = size / (max(width, height) * 1.5)
    points = landmarks(face)

    if mode == "insightface":
        matrix = np.asarray([[scale, 0.0, size / 2.0 - cx * scale], [0.0, scale, size / 2.0 - cy * scale],], dtype = np.float32)

        return cv2.warpAffine(frame, matrix, (size, size), borderValue = 0.0)

    if mode == "letterbox":
        ix1 = max(0, int(x1))
        iy1 = max(0, int(y1))
        ix2 = min(frame.shape[1], int(x2))
        iy2 = min(frame.shape[0], int(y2))
        crop = frame[iy1:iy2, ix1:ix2]

        if crop.size == 0:
            return np.zeros((size, size, 3), dtype = frame.dtype)
        ratio = min(size / crop.shape[0], size / crop.shape[1])
        new_w = max(1, round(crop.shape[1] * ratio))
        new_h = max(1, round(crop.shape[0] * ratio))
        resized = cv2.resize(crop, (new_w, new_h), interpolation = cv2.INTER_LINEAR)
        canvas  = np.zeros((size, size, 3), dtype = frame.dtype)
        left    = (size - new_w) // 2
        top     = (size - new_h) // 2
        canvas[top:top + new_h, left:left + new_w] = resized

        return canvas

    if mode == "arcface" and points is not None:
        ratio = size / 112.0
        dst   = ARCFACE_DST * ratio
        shift = 8.0 * ratio
        dst[:, 0] += shift
        dst[:, 1] += shift
        matrix, _ = cv2.estimateAffinePartial2D(points, dst)

        if matrix is not None:

            return cv2.warpAffine(frame, matrix, (size, size), borderValue=0.0)

    if mode == "roll" and points is not None:
        left_eye, right_eye = points[0], points[1]

        if abs(right_eye[0] - left_eye[0]) > 1.0:
            roll = math.degrees(math.atan2(float(right_eye[1] - left_eye[1]), float(right_eye[0] - left_eye[0])))
            roll = max(-30.0, min(30.0, roll))

            if abs(roll) > 1.0:
                matrix = cv2.getRotationMatrix2D((cx, cy), roll, scale)
                matrix[0, 2] += size / 2.0 - cx
                matrix[1, 2] += size / 2.0 - cy

                return cv2.warpAffine(frame, matrix, (size, size), borderValue = 0.0)

    matrix = np.asarray([[scale, 0.0, size / 2.0 - cx * scale], [0.0, scale, size / 2.0 - cy * scale],], dtype = np.float32)

    return cv2.warpAffine(frame, matrix, (size, size), borderValue=0.0)


def prior_boxes(size: int = FACE_DETECTOR_SIZE) -> np.ndarray:
    anchors   = []
    min_sizes = ((16, 32), (64, 128), (256, 512))
    steps     = (8, 16, 32)

    for level, step in enumerate(steps):
        rows = math.ceil(size / step)
        cols = math.ceil(size / step)

        for row, col in product(range(rows), range(cols)):

            for min_size in min_sizes[level]:
                anchors.append(((col + 0.5) * step / size, (row + 0.5) * step / size, min_size / size, min_size / size))

    return np.asarray(anchors, dtype=np.float32)


def decode_boxes(priors: np.ndarray, loc: np.ndarray) -> np.ndarray:
    variances = np.asarray((0.1, 0.2), dtype=np.float32)
    boxes     = np.concatenate((priors[:,: 2] + loc[:,: 2] * variances[0] * priors[:, 2:], priors[:, 2:] * np.exp(loc[:, 2:] * variances[1]),), axis = 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]

    return boxes


def decode_landmarks(priors: np.ndarray, raw: np.ndarray) -> np.ndarray:
    points = []

    for i in range(5):
        points.append(priors[:,: 2] + raw[:, 2 * i: 2 * i + 2] * 0.1 * priors[:, 2:])

    return np.concatenate(points, axis=1)


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)
    order = scores.argsort()[::-1]
    kept: list[int] = []
    while order.size:
        i = int(order[0])
        kept.append(i)

        if order.size == 1:
            break
        rest    = order[1:]
        xx1     = np.maximum(x1[i], x1[rest])
        yy1     = np.maximum(y1[i], y1[rest])
        xx2     = np.minimum(x2[i], x2[rest])
        yy2     = np.minimum(y2[i], y2[rest])
        inter   = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1)
        overlap = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-6)
        order   = rest[overlap <= threshold]

    return kept
