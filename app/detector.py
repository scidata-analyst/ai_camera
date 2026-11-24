"""Human-only detector: YOLO 'person' class + RetinaFace, no non-human output."""
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

from .smoother import CountSmoother, FaceAttributeSmoother
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
        self.conf = conf
        self.imgsz = imgsz
        self.model_name = model_name
        self.face_min_size = int(face_min_size)
        # NPU-only: CPU/auto/ONNX fallback removed. Missing RKNN => fail fast.
        self.backend = os.getenv("YOLO_BACKEND", "rknn").lower()
        if self.backend not in ("rknn", ""):
            raise RuntimeError(
                f"YOLO_BACKEND={self.backend!r} is disabled in NPU-only mode")
        self.backend = "rknn"
        self.rknn_path = os.getenv("RKNN_MODEL", "/app/models/yolo11n-rk3576.rknn")
        self.model = None
        # NPU device log: /dev/rknpu (NPU) + /dev/dri/renderD129 (Mali GPU)
        try:
            from .accel import hw_summary
            print(f"[hw] {hw_summary()} rknn_model={self.rknn_path} "
                  f"exists={os.path.exists(self.rknn_path)}")
        except Exception:
            pass
        from .accel import rknn_input_summary, rknn_load
        self.rknn, core = rknn_load(self.rknn_path, "yolo")
        print(f"[yolo] NPU input: {rknn_input_summary(self.rknn)}")
        print(f"[yolo] NPU backend active: {self.rknn_path} core={core}")
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
        # 1-sec window-e sob face-er age/gender prediction average kore dey.
        self.attr_smoother = FaceAttributeSmoother(window_sec=window_sec)
        self._t_prev = time.time()
        self._fps_smooth = 0.0
        # LAG FIX: YOLO proti frame e na chaliye reuse koro.
        # DETECT_EVERY=2 mane 2 bar er moddhe 1 bar YOLO, majhe box reuse.
        # RK3576 CPU te ~2x infer fps bare.
        self.detect_every = max(1, int(os.getenv("DETECT_EVERY", "2")))
        # 0 = no cap. Frame-e joto face thakbe sob process korbo.
        self.max_faces = max(0, int(os.getenv("MAX_FACES", "0")))
        # False-positive guard: a 'person' box must clear its own threshold.
        # Objects misread as person usually sit between 0.5 and 0.65.
        self.person_conf = float(os.getenv("PERSON_CONF", "0.65"))
        self.person_min_h = int(os.getenv("PERSON_MIN_H", "0"))
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
        if self.rknn is None:
            raise RuntimeError("[yolo] NPU backend unavailable — CPU disabled")
        return self._run_rknn(frame)

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
        # Model er compiled input NHWC/FLOAT16. Ekhon 4D NHWC pathai — age
        # NCHW pathailo Lite2 image-ke ulto-pulte podhe, class score garbage
        # hoto ar sob kichu 'person' dekha jeto.
        nhwc = _np.ascontiguousarray(rgb[_np.newaxis, ...], dtype=_np.float32)
        with self._lock:
            outs = rknn_infer(self.rknn, nhwc, "nhwc", "float32")
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
        # Strict human-only: COCO class 0 = 'person'. Other class-er score
        # dekheo detect korbo na, label-o dekhabo na.
        person_scores = scores[:, 0]
        if self._debug and not getattr(self, "_rknn_debug_done", False):
            self._rknn_debug_done = True
            print(f"[yolo-debug] max person score="
                  f"{float(person_scores.max()) if len(person_scores) else 0.0:.3f}")
        keep = person_scores >= self.person_conf
        boxes_cxcywh = boxes_cxcywh[keep]
        confs = person_scores[keep]
        cls_ids = _np.zeros(len(confs), dtype=_np.int64)
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
                    self.conf, 0.45)
                picked = [int(x) for x in _np.asarray(picked).reshape(-1)]
            except Exception:
                picked = list(range(len(ix)))
            kept.update(int(ix[p]) for p in picked)
        # Kono count cap nai — frame-e joto object thakbe sob return korbo.
        kept = sorted(kept, key=lambda i: -float(confs[i]))
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
            """Strict human-only: non-human kisui passe na."""
            if d["kind"] != "human":
                return False
            x1, y1, x2, y2 = d["box"]
            return (d["conf"] >= self.person_conf and
                    (y2 - y1) >= self.person_min_h)

        if run_yolo:
            _t0 = time.perf_counter()
            results = self._run_yolo(frame)
            self.timing["yolo_ms"] = round(
                (time.perf_counter() - _t0) * 1000, 1)
            for box in results.boxes:
                cls_id = int(box.cls[0])
                if cls_id != 0:
                    continue  # strictly human-only
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                d = {"kind": "human", "class": "person",
                     "conf": round(conf, 3), "box": [x1, y1, x2, y2]}
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
                    # MAX_FACES=0 (default) means no cap; >0 only if explicitly set.
                    if self.max_faces > 0 and len(faces) > self.max_faces:
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
                    # 1-sec window-e sob fresh prediction joma koro.
                    self.attr_smoother.add(faces)
                    if self.headpose is not None:
                        faces = self.headpose.fill_from_cache(faces)
                        self._headpose_submit(source_frame, faces)
                    self._last_faces_cache = [dict(f) for f in faces]
                else:
                    faces = [dict(f) for f in self._last_faces_cache]
                # 1-sec average age/gender, sticky label (no flicker/disappear).
                self.attr_smoother.fill(faces)
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
        # Non-human detect/label — duto-i choles na.
        non_human_n = 0

        def _face_for_human(d: Dict):
            """Human box-er moddhe thaka (average-smoothed) face khujo."""
            x1, y1, x2, y2 = d["box"]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            best, best_area = None, None
            for f in faces:
                fx1, fy1, fx2, fy2 = f["box"]
                if fx1 <= cx <= fx2 and fy1 <= cy <= fy2:
                    area = (fx2 - fx1) * (fy2 - fy1)
                    if best_area is None or area < best_area:
                        best, best_area = f, area
            return best

        # Boxes draw kora hoy face/check-er por, jate bade dewa box dekha na jay.
        for d in detections:
            if d["kind"] != "human":
                continue
            x1, y1, x2, y2 = d["box"]
            face = _face_for_human(d)
            if face is not None:
                gender = face.get("gender") or "Unknown"
                age = face.get("age", 0)
            else:
                # Onno manush-er age/gender copy korbo na.
                gender, age = "Unknown", 0
            label = f"Human {gender} {age}y {d['conf']:.2f}"
            color = (0, 255, 0)  # green
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
