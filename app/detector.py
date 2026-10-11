"""YOLOv11 based Human vs Non-Human detector."""
import os
import threading
import time
from typing import List, Dict

import cv2
import numpy as np

# LAG FIX: CPU thread fight komao (torch + onnx + cv2)
try:
    _NTH = int(os.getenv("TORCH_THREADS", str(min(4, (os.cpu_count() or 4)))))
    cv2.setNumThreads(max(1, min(4, _NTH)))
except Exception:
    pass

from .smoother import CountSmoother
from .face_detector import YuNetFace
from .mivolo import MiVOLOAgeGender
from .headpose import HeadPoseEstimator


class _RknnBox:
    """Ultralytics box adapter: box.cls[0] / box.conf[0] / box.xyxy[0]."""
    def __init__(self, cls_id: int, conf: float, xyxy):
        self.cls = np.array([cls_id], dtype=np.float32)
        self.conf = np.array([conf], dtype=np.float32)
        self.xyxy = np.array([xyxy], dtype=np.float32)


class _RknnResult:
    def __init__(self, boxes):
        self.boxes = boxes


class HumanDetector:
    """YOLO Human + YuNet Face + MiVOLO age/gender + WHENet looking."""

    def __init__(self, model_name: str = "yolo11n.pt", conf: float = 0.5, imgsz: int = 640,
                 window_sec: float = 1.0, face_enable: bool = True,
                 mivolo_enable: bool = True, headpose_enable: bool = True,
                 face_min_size: int = 60):
        from ultralytics import YOLO

        self.conf = conf
        self.imgsz = imgsz
        self.model_name = model_name
        self.face_min_size = int(face_min_size)
        # ---- NPU backend: YOLO_BACKEND=auto|cpu|rknn ----
        # auto = RKNN_MODEL file + rknnlite thakle NPU, naile CPU fallback.
        # .rknn akhono na banale safe CPU te cholbe, crash korbe na.
        self.backend = os.getenv("YOLO_BACKEND", "auto").lower()
        self.rknn_path = os.getenv("RKNN_MODEL", "/app/models/yolo11n-rk3576.rknn")
        self.rknn = None
        if self.backend in ("auto", "rknn") and os.path.exists(self.rknn_path):
            try:
                from rknnlite.api import RKNNLite
                core = os.getenv("RKNPU_CORE", "AUTO")
                self.rknn = RKNNLite()
                # NPU core: AUTO default; RK3576 te 0/1/2 force kora jay
                load_kw = {}
                if core.upper() not in ("AUTO", ""):
                    try:
                        load_kw["core_mask"] = int(core)
                    except ValueError:
                        pass
                ret = self.rknn.load_rknn(self.rknn_path)
                if ret != 0:
                    raise RuntimeError(f"load_rknn ret={ret}")
                ret = self.rknn.init_runtime(**load_kw) if load_kw else self.rknn.init_runtime()
                if ret != 0:
                    raise RuntimeError(f"init_runtime ret={ret}")
                print(f"[yolo] NPU backend active: {self.rknn_path} core={core}")
            except Exception as e:
                print(f"[yolo] NPU init fail, CPU fallback: {e}")
                self.rknn = None
        if self.rknn is not None:
            self.backend = "rknn"
            self.model = None
            self.names = {i: n for i, n in enumerate(
                ["person", "bicycle", "car", "motorcycle", "airplane", "bus",
                 "train", "truck", "boat", "traffic light", "fire hydrant",
                 "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                 "horse", "sheep", "cow", "elephant", "bear", "zebra",
                 "giraffe", "backpack", "umbrella", "handbag", "tie",
                 "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                 "kite", "baseball bat", "baseball glove", "skateboard",
                 "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                 "fork", "knife", "spoon", "bowl", "banana", "apple",
                 "sandwich", "orange", "broccoli", "carrot", "hot dog",
                 "pizza", "donut", "cake", "chair", "couch", "potted plant",
                 "bed", "dining table", "toilet", "tv", "laptop", "mouse",
                 "remote", "keyboard", "cell phone", "microwave", "oven",
                 "toaster", "sink", "refrigerator", "book", "clock", "vase",
                 "scissors", "teddy bear", "hair drier", "toothbrush"])}
        else:
            self.backend = "cpu"
            # lazy download on first run; Docker build e cache kora jabe
            self.model = YOLO(model_name)
            self.names = self.model.names  # {0: 'person', ...}
        self._lock = threading.Lock()
        self.last_counts = {"human": 0, "non_human": 0, "face": 0,
                            "male": 0, "female": 0, "fps": 0.0}
        self.last_faces = []
        self.face = YuNetFace() if face_enable else None
        self.face_enabled = self.face is not None and self.face.detector is not None
        self.mivolo = MiVOLOAgeGender() if (mivolo_enable and self.face_enabled) else None
        self.mivolo_enabled = self.mivolo is not None and self.mivolo.loaded
        self.headpose = HeadPoseEstimator() if (headpose_enable and self.face_enabled) else None
        self.headpose_enabled = self.headpose is not None and self.headpose.loaded
        self.smoother = CountSmoother(window_sec=window_sec)
        self.smoothed = self.smoother.current
        self._t_prev = time.time()
        self._fps_smooth = 0.0
        # LAG FIX: YOLO proti frame e na chaliye reuse koro.
        # DETECT_EVERY=2 mane 2 bar er moddhe 1 bar YOLO, majhe box reuse.
        # RK3576 CPU te ~2x infer fps bare.
        self.detect_every = max(1, int(os.getenv("DETECT_EVERY", "2")))
        self.max_faces = max(1, int(os.getenv("MAX_FACES", "3")))
        self._frame_no = 0
        self._last_dets: List[Dict] = []

    def _run_yolo(self, frame: np.ndarray):
        if self.rknn is not None:
            return self._run_rknn(frame)
        with self._lock:
            results = self.model.predict(
                source=frame, conf=self.conf, imgsz=self.imgsz,
                verbose=False, half=False, augment=False,
                agnostic_nms=True, max_det=20,
            )[0]
        return results

    # ---------- RK3576 NPU (RKNNLite) path ----------
    @staticmethod
    def _letterbox(img: np.ndarray, size: int):
        """Ultralytics-style letterbox -> square size x size + undo params."""
        h, w = img.shape[:2]
        r = min(size / h, size / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top = (size - nh) // 2
        left = (size - nw) // 2
        out = np.full((size, size, 3), 114, dtype=np.uint8)
        out[top:top + nh, left:left + nw] = resized
        return out, r, left, top

    def _run_rknn(self, frame: np.ndarray):
        """NPU infer + Python NMS. Output shape auto-handle (84xN / Nx84)."""
        import numpy as _np
        H, W = frame.shape[:2]
        img, ratio, pad_x, pad_y = self._letterbox(frame, self.imgsz)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(_np.float32) / 255.0
        nchw = _np.transpose(rgb, (2, 0, 1))[_np.newaxis, ...]
        with self._lock:
            outs = self.rknn.inference(inputs=[nchw])
        pred = None
        for o in outs:
            o = _np.asarray(o)
            if o.ndim == 3 and (o.shape[1] == 84 or o.shape[2] == 84):
                pred = o
                break
        if pred is None:
            if not getattr(self, "_rknn_warned", False):
                print(f"[yolo] NPU unexpected outputs: "
                      f"{[tuple(_np.asarray(o).shape) for o in outs]} "
                      f"-> empty (shape janiye pathao, fix kore debo)")
                self._rknn_warned = True
            return _RknnResult([])
        if pred.shape[1] == 84:
            pred = _np.transpose(pred[0], (1, 0))  # (N,84)
        else:
            pred = pred[0]  # (N,84)
        boxes_cxcywh = pred[:, :4]
        scores = pred[:, 4:]
        cls_ids = _np.argmax(scores, axis=1)
        confs = scores[_np.arange(len(scores)), cls_ids]
        keep = confs >= self.conf
        # sudhu person (0) + non-human box — sob class NMS er age filter na,
        # conf filter e enough; agnostic NMS er moto 1 bar NMS
        boxes_cxcywh, cls_ids, confs = boxes_cxcywh[keep], cls_ids[keep], confs[keep]
        if len(confs) == 0:
            return _RknnResult([])
        # letterbox undo -> original frame coords
        cx, cy, bw, bh = boxes_cxcywh.T
        x1 = ((cx - bw / 2 - pad_x) / ratio).clip(0, W)
        y1 = ((cy - bh / 2 - pad_y) / ratio).clip(0, H)
        x2 = ((cx + bw / 2 - pad_x) / ratio).clip(0, W)
        y2 = ((cy + bh / 2 - pad_y) / ratio).clip(0, H)
        ltrb = _np.stack([x1, y1, x2, y2], axis=1).tolist()
        try:
            idxs = cv2.dnn.NMSBoxes(ltrb, confs.tolist(),
                                    self.conf, 0.45, top_k=20)
            idxs = set(int(i) for arr in _np.asarray(idxs).reshape(-1)
                       for i in ([arr] if _np.ndim(arr) == 0 else arr))
        except Exception:
            idxs = set(range(len(confs)))
        out = [_RknnBox(int(cls_ids[i]), float(confs[i]),
                        [int(ltrb[i][0]), int(ltrb[i][1]),
                         int(ltrb[i][2]), int(ltrb[i][3])])
               for i in sorted(idxs)[:20]]
        return _RknnResult(out)

    def infer(self, frame: np.ndarray) -> tuple[np.ndarray, List[Dict]]:
        """Run detection, draw boxes, return (annotated_frame, detections)."""
        self._frame_no += 1
        # LAG FIX: proti N frame e 1 bar YOLO, majhe ager box reuse.
        # Face/YuNet proti bar chole (sosta), YOLO skip holeo video atke na.
        run_yolo = (self._frame_no % self.detect_every == 1) or not self._last_dets

        detections: List[Dict] = []
        human_n = 0
        non_human_n = 0

        if run_yolo:
            results = self._run_yolo(frame)
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                coco_name = self.names.get(cls_id, str(cls_id))

                if cls_id == 0:  # person
                    kind = "human"
                    human_n += 1
                else:
                    kind = "non_human"
                    non_human_n += 1

                detections.append(
                    {"kind": kind, "class": coco_name, "conf": round(conf, 3),
                     "box": [x1, y1, x2, y2]}
                )
            self._last_dets = [dict(d) for d in detections]
        else:
            detections = [dict(d) for d in self._last_dets]
            human_n = sum(1 for d in detections if d["kind"] == "human")
            non_human_n = sum(1 for d in detections if d["kind"] == "non_human")

        for d in detections:
            x1, y1, x2, y2 = d["box"]
            if d["kind"] == "human":
                label = f"Human {d['conf']:.2f}"
                color = (0, 255, 0)  # green
            else:
                label = f"Non-Human:{d['class']} {d['conf']:.2f}"
                color = (0, 165, 255)  # orange

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # YuNet face -> WHENet looking -> MiVOLO age/gender (body = YOLO box).
        # looking age ber hoy, jate blurry/side frame e age refresh off thake.
        faces = []
        if self.face is not None:
            try:
                faces = self.face.detect(frame)
                # LAG FIX: sob face na — sobcheye boro MAX_FACES ta.
                # choto/durer face e WHENet+MiVOLO chalano CPU noshto.
                if len(faces) > self.max_faces:
                    faces = sorted(
                        faces,
                        key=lambda f: -((f["box"][2] - f["box"][0]) *
                                        (f["box"][3] - f["box"][1])),
                    )[:self.max_faces]
                if self.headpose is not None and faces:
                    faces = self.headpose.predict_frame(frame, faces)
                # Gender+Age: sudhu boro/clear/frontal face e refresh
                if self.mivolo is not None and faces:
                    bodies = [d["box"] for d in detections if d["kind"] == "human"]
                    faces = self.mivolo.predict_frame(
                        frame, faces, bodies, min_side=self.face_min_size)
                YuNetFace.draw(frame, faces)
            except Exception as e:
                print(f"[face] infer fail: {e}")
                faces = []
        face_n = len(faces)
        male_n = sum(1 for f in faces if f.get("gender") == "Male")
        female_n = sum(1 for f in faces if f.get("gender") == "Female")
        look_n = sum(1 for f in faces if f.get("looking"))
        self.last_faces = faces

        # FPS + 1-sec smoothed header (mean/median/mode)
        now = time.time()
        inst = 1.0 / max(now - self._t_prev, 1e-6)
        self._t_prev = now
        self._fps_smooth = self._fps_smooth * 0.9 + inst * 0.1

        self.smoothed = self.smoother.add(human_n, non_human_n, face_n,
                                            male_n, female_n)
        sh = self.smoothed["stable_human"]
        sn = self.smoothed["stable_non_human"]
        sf = self.smoothed.get("stable_face", face_n)
        sm = self.smoothed.get("stable_male", male_n)
        sw = self.smoothed.get("stable_female", female_n)
        mh = self.smoothed["human"]["mean"]

        header = (f"H:{human_n}[{sh}] F:{face_n}[{sf}] "
                  f"M:{male_n}[{sm}] W:{female_n}[{sw}] Look:{look_n} FPS:{self._fps_smooth:.1f}")
        sub = (f"1s avg mean:{mh} median:{self.smoothed['human']['median']} "
               f"mode:{self.smoothed['human']['mode']}")
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 56), (0, 0, 0), -1)
        cv2.putText(frame, header, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, sub, (10, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        self.last_counts = {
            "human": human_n, "non_human": non_human_n, "face": face_n,
            "male": male_n, "female": female_n, "looking": look_n,
            "fps": round(self._fps_smooth, 1),
        }
        return frame, detections
