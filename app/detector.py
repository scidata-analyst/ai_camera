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
from .retinaface import RetinaFace
from .age_gender import AgeGender
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
    """YOLO humans/objects + RetinaFace + age/gender + WHENet head pose."""

    def __init__(self, model_name: str = "yolo11n.pt", conf: float = 0.5, imgsz: int = 640,
                 window_sec: float = 1.0, face_enable: bool = True,
                 headpose_enable: bool = True, face_min_size: int = 60):
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
        # NPU device log: /dev/rknpu (NPU) + /dev/dri/renderD129 (Mali GPU)
        try:
            from .accel import hw_summary
            print(f"[hw] {hw_summary()} rknn_model={self.rknn_path} "
                  f"exists={os.path.exists(self.rknn_path)}")
        except Exception:
            pass
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
                try:
                    from .accel import rknn_input_summary
                    print(f"[yolo] NPU input: {rknn_input_summary(self.rknn)}")
                except Exception:
                    pass
                print(f"[yolo] NPU backend active: {self.rknn_path} core={core}")
            except Exception as e:
                print(f"[yolo] NPU init fail, CPU fallback: {e}")
                self.rknn = None
        elif self.backend in ("auto", "rknn"):
            print(f"[yolo] RKNN model missing, CPU fallback: {self.rknn_path}")
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
        self.face = RetinaFace() if face_enable else None
        self.face_enabled = self.face is not None and self.face.loaded
        self.age_gender = AgeGender() if (face_enable and self.face_enabled) else None
        self.age_gender_enabled = self.age_gender is not None and self.age_gender.loaded
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
        # False-positive guard: a 'person' box must clear its own threshold.
        # Objects misread as person usually sit between 0.5 and 0.65.
        self.person_conf = float(os.getenv("PERSON_CONF", "0.65"))
        self.person_min_h = int(os.getenv("PERSON_MIN_H", "40"))
        self.person_require_face = os.getenv("PERSON_REQUIRE_FACE", "0") == "1"
        # Never reuse a stale box longer than this (seconds).
        self.det_stale_sec = float(os.getenv("DET_STALE_SEC", "0.2"))
        self._last_dets_t = 0.0
        self._debug = os.getenv("YOLO_DEBUG", "0") == "1"
        # SPEED: RetinaFace proti infer e na chaliye N bar por por (box reuse).
        # FACE_EVERY=2 mane YOLO skip er sathe face o half rate -> CPU ~30% save.
        self.face_every = max(1, int(os.getenv("FACE_EVERY", "2")))
        self._face_no = 0
        self._last_faces_cache = []
        # Age/gender is a tiny single-face model and runs alongside face detect.
        # Head-pose remains on a latest-only worker so it cannot stall the stream.
        self._hp_lock = threading.Lock()
        self._hp_pending = None
        # Diagnose stage latency through /api/stats.
        self._bg_last_ms = 0.0
        self.timing = {"yolo_ms": 0.0, "face_ms": 0.0, "age_gender_ms": 0.0,
                       "infer_ms": 0.0}
        if self.headpose_enabled:
            threading.Thread(target=self._bg_loop, daemon=True).start()
        self._frame_no = 0
        self._last_dets: List[Dict] = []

    def _headpose_submit(self, frame: np.ndarray, faces):
        """Keep only the newest head-pose job; stale frames are discarded."""
        if self.headpose is None or not faces:
            return
        fcs = [{"box": list(f["box"]), "landmarks": f.get("landmarks"),
                "looking": f.get("looking")}
               for f in faces]
        with self._hp_lock:
            self._hp_pending = (frame.copy(), fcs)

    def _bg_loop(self):
        """Run only WHENet head pose on the newest face set."""
        while True:
            try:
                with self._hp_lock:
                    job = self._hp_pending
                    self._hp_pending = None
                if job is None:
                    time.sleep(0.05)
                    continue
                frame, fcs = job
                _bt0 = time.perf_counter()
                try:
                    fcs = self.headpose.predict_frame(frame, fcs)
                    self.headpose.remember(fcs)
                except Exception as e:
                    print(f"[headpose-bg] {e}")
                self._bg_last_ms = round(
                    (time.perf_counter() - _bt0) * 1000, 1)
            except Exception as e:
                print(f"[bg] {e}")
                time.sleep(0.2)

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
        """NPU infer + class-wise NMS. Output shape auto-handle (84xN / Nx84)."""
        import numpy as _np
        from .accel import rknn_infer
        H, W = frame.shape[:2]
        img, ratio, pad_x, pad_y = self._letterbox(frame, self.imgsz)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(_np.float32) / 255.0
        # Model er compiled input NHWC/FLOAT16; amra pathai NCHW float32.
        # data_format na dile Lite2 default NHWC dhoro → image ulto-pulte
        # podhe → class score garbage → sob kichu 'person' hoy.
        nchw = _np.ascontiguousarray(
            _np.transpose(rgb, (2, 0, 1))[_np.newaxis, ...])
        with self._lock:
            outs = rknn_infer(self.rknn, nchw, "nchw", "float32")
        if not outs:
            if not getattr(self, "_rknn_warned", False):
                print("[yolo] NPU returned no outputs (inference fail)")
                self._rknn_warned = True
            return _RknnResult([])
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
        # Ultralytics YOLO11 export already applies sigmoid to class scores.
        # Applying it again would inflate every confidence to >= 0.5.
        scores = pred[:, 4:]
        cls_ids = _np.argmax(scores, axis=1)
        confs = scores[_np.arange(len(scores)), cls_ids]
        if self._debug and not getattr(self, "_rknn_debug_done", False):
            self._rknn_debug_done = True
            best = pred[int(_np.argmax(confs))]
            order = _np.argsort(best[4:])[-5:][::-1]
            print("[yolo-debug] top classes: " + ", ".join(
                f"{self.names.get(int(i), i)}={float(best[4 + int(i)]):.3f}"
                for i in order))
        # Person-get alada (beshi) threshold — object-ke human bananor mota
        # 0.5-0.65 score gulo etay ukriye jay.
        needed = _np.where(cls_ids == 0, self.person_conf, self.conf)
        keep = confs >= needed
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
        # OpenCV NMSBoxes expects [x, y, width, height], while ltrb is
        # [x1, y1, x2, y2]. Passing ltrb directly corrupts overlap tests.
        nms_boxes = [
            [int(box[0]), int(box[1]),
             max(1, int(box[2] - box[0])),
             max(1, int(box[3] - box[1]))]
            for box in ltrb
        ]
        # Class-wise NMS: ek class-er box onno class-er box diye chapai jay na.
        kept: set[int] = set()
        for cid in _np.unique(cls_ids):
            ix = _np.flatnonzero(cls_ids == cid)
            try:
                picked = cv2.dnn.NMSBoxes(
                    [nms_boxes[i] for i in ix], confs[ix].tolist(),
                    self.conf, 0.45, top_k=10)
                picked = [int(x) for x in _np.asarray(picked).reshape(-1)]
            except Exception:
                picked = list(range(len(ix)))
            kept.update(int(ix[p]) for p in picked)
        kept = sorted(kept, key=lambda i: -float(confs[i]))[:20]
        out = [_RknnBox(int(cls_ids[i]), float(confs[i]),
                        [int(ltrb[i][0]), int(ltrb[i][1]),
                         int(ltrb[i][2]), int(ltrb[i][3])])
               for i in kept]
        return _RknnResult(out)

    def infer(self, frame: np.ndarray) -> tuple[np.ndarray, List[Dict]]:
        """Run detection, draw boxes, return (annotated_frame, detections)."""
        # Keep model inputs free of the boxes/labels drawn below. In particular,
        # face alignment and age/gender inference are sensitive to overlays.
        source_frame = frame.copy()
        self._frame_no += 1
        _t_all = time.perf_counter()
        # LAG FIX: proti N frame e 1 bar YOLO, majhe ager box reuse.
        # Face/RetinaFace periodically runs while YOLO boxes are reused.
        # Phase stagger: YOLO even frame e, face detector odd frame e — vari 2 kaj
        # eki frame e porle spike hoto. (==0 phase, EVERY=1 eo thik chole;
        # ager ==1 logic e EVERY=1 dile kokhono re-run hoto na.)
        run_yolo = (self._frame_no % self.detect_every == 0) or not self._last_dets
        # Purono box beshi din reuse korle camera naoleo 'human' dekha jay.
        if not run_yolo and self._last_dets_t and \
                time.time() - self._last_dets_t > self.det_stale_sec:
            self._last_dets = []
            run_yolo = True

        detections: List[Dict] = []

        def _keep(d: Dict) -> bool:
            """Person-get alada threshold + choto box filter (duto backend-e)."""
            x1, y1, x2, y2 = d["box"]
            if d["kind"] == "human":
                if d["conf"] < self.person_conf or (y2 - y1) < self.person_min_h:
                    return False
            return d["conf"] >= self.conf

        if run_yolo:
            _t0 = time.perf_counter()
            results = self._run_yolo(frame)
            self.timing["yolo_ms"] = round(
                (time.perf_counter() - _t0) * 1000, 1)
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                coco_name = self.names.get(cls_id, str(cls_id))
                d = {"kind": "human" if cls_id == 0 else "non_human",
                     "class": coco_name, "conf": round(conf, 3),
                     "box": [x1, y1, x2, y2]}
                if _keep(d):
                    detections.append(d)
            self._last_dets = [dict(d) for d in detections]
            self._last_dets_t = time.time()
        else:
            detections = [dict(d) for d in self._last_dets if _keep(d)]

        # RetinaFace finds faces; the small age/gender head predicts inline.
        # FACE_EVERY controls whether the previous face results are reused.
        faces = []
        if self.face is not None:
            try:
                self._face_no += 1
                # YOLO phase 0 (even), face phase (EVERY-1, odd) — spike vag hoy
                run_face = (self._face_no % self.face_every == self.face_every - 1) \
                    or not self._last_faces_cache
                if run_face:
                    _t0 = time.perf_counter()
                    faces = self.face.detect(source_frame)
                    self.timing["face_ms"] = round(
                        (time.perf_counter() - _t0) * 1000, 1)
                    # Limit small distant faces to keep camera inference bounded.
                    if len(faces) > self.max_faces:
                        faces = sorted(
                            faces,
                            key=lambda f: -((f["box"][2] - f["box"][0]) *
                                            (f["box"][3] - f["box"][1])),
                        )[:self.max_faces]
                    _t1 = time.perf_counter()
                    if self.age_gender is not None:
                        for face in faces:
                            x1, y1, x2, y2 = face["box"]
                            if min(x2 - x1, y2 - y1) < self.face_min_size:
                                face.update({"gender": "Unknown", "gconf": 0.0,
                                             "age": 0})
                            else:
                                self.age_gender.predict(source_frame, face)
                    self.timing["age_gender_ms"] = round(
                        (time.perf_counter() - _t1) * 1000, 1)
                    if self.headpose is not None:
                        faces = self.headpose.fill_from_cache(faces)
                        self._headpose_submit(source_frame, faces)
                    self._last_faces_cache = [dict(f) for f in faces]
                else:
                    faces = [dict(f) for f in self._last_faces_cache]
                RetinaFace.draw(frame, faces)
            except Exception as e:
                print(f"[face] infer fail: {e}")
                faces = []

        # Optional cross-check: PERSON_REQUIRE_FACE=1 mantre face na thaka
        # 'human' box gulo bade dewa hoy (chairs/bags ke human bananor roodh).
        if self.person_require_face:
            def _has_face(d: Dict) -> bool:
                x1, y1, x2, y2 = d["box"]
                return any(x1 <= f["box"][0] <= x2 and y1 <= f["box"][1] <= y2
                           for f in faces)
            before = len(detections)
            detections = [d for d in detections
                          if d["kind"] != "human" or _has_face(d)]
            if len(detections) != before:
                self._last_dets = [dict(d) for d in detections]

        human_n = sum(1 for d in detections if d["kind"] == "human")
        non_human_n = sum(1 for d in detections if d["kind"] == "non_human")

        # Boxes draw kora hoy face/check-er por, jate bade dewa box dekha na jay.
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

        self.timing["infer_ms"] = round(
            (time.perf_counter() - _t_all) * 1000, 1)
        self.last_counts = {
            "human": human_n, "non_human": non_human_n, "face": face_n,
            "male": male_n, "female": female_n, "looking": look_n,
            "fps": round(self._fps_smooth, 1),
            "t_yolo_ms": self.timing["yolo_ms"],
            "t_face_ms": self.timing["face_ms"],
            "t_age_gender_ms": self.timing["age_gender_ms"],
            "t_infer_ms": self.timing["infer_ms"],
            "t_bg_ms": self._bg_last_ms,
            "bg_pending": self._hp_pending is not None,
        }
        return frame, detections
