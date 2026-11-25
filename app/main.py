from __future__ import annotations

import queue
import threading
import time
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image, ImageDraw, ImageFont

from .config.settings import Settings
from .domain.geometry import iou, sharpness
from .infrastructure.rknn.genderage_estimator import RknnGenderAgeEstimator
from .infrastructure.rknn.retinaface_detector import RknnFaceDetector
from .infrastructure.rknn.yolo_detector import RknnPersonDetector
from .interfaces.auth import require_token

WEB_DIR     = Path(__file__).resolve().parent / "interfaces" / "web"
INDEX_PATH  = WEB_DIR / "index.html"
FONT_PATH   = WEB_DIR / "UbuntuSans.ttf"
_FONT_CACHE  = {}
_PLACEHOLDER = None


def _placeholder() -> np.ndarray:
    global _PLACEHOLDER

    if _PLACEHOLDER is None:
        image = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
        draw  = ImageDraw.Draw(image)
        draw.text((30, 220), "Waiting for camera frames...", font=_font(26), fill=(200, 210, 225))
        _PLACEHOLDER = np.asarray(image).copy()

    return _PLACEHOLDER


def _font(size: int) -> ImageFont.FreeTypeFont:

    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = ImageFont.truetype(str(FONT_PATH), size)

    return _FONT_CACHE[size]


def _mode(values: list, most_recent_on_tie: bool = True):
    if not values:

        return None
    counts = Counter(values)
    top    = max(counts.values())
    tied   = [value for value, count in counts.items() if count == top]

    if len(tied) == 1:

        return tied[0]

    if not most_recent_on_tie:

        return tied[0]
    last_seen = {}

    for index, value in enumerate(values):
        last_seen[value] = index

    return max(tied, key=lambda value: last_seen[value])


class _Tracker:
    def __init__(self):
        self._next  = 1
        self._boxes = {}

    def match(self, boxes: list[list[int]]) -> list[tuple[int, list[int]]]:

        if not boxes:

            return []
        used    = set()
        matched = []

        for box in boxes:
            best, best_iou = None, 0.0

            for person_id, previous in self._boxes.items():

                if person_id in used:

                    continue
                overlap = iou(box, previous)

                if overlap > best_iou:
                    best, best_iou = person_id, overlap

            if best is not None and best_iou >= 0.3:
                person_id = best
                used.add(person_id)
                self._boxes[person_id] = box
            else:
                person_id = self._next
                self._next += 1
                self._boxes[person_id] = box
                used.add(person_id)
            matched.append((person_id, box))

        self._boxes = {key: self._boxes[key] for key in used if key in self._boxes}

        return matched


class Pipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.person   = RknnPersonDetector(settings)
        self.face     = RknnFaceDetector(settings) if settings.face_enable else None
        face_loaded   = self.face is not None and self.face.describe().get("loaded", False)
        self.age      = RknnGenderAgeEstimator(settings) if face_loaded else None
        self.tracker  = _Tracker()
        self.incoming = queue.Queue()
        self.stop     = threading.Event()
        self._lock    = threading.Lock()
        self.preview      = None
        self.overlay_people = []
        self.overlay_faces  = []
        self.result       = {}
        self.last_error   = None
        self.timings      = {"yolo": 0.0, "face": 0.0, "age": 0.0}
        self.top_frame    = 0
        self.camera_info  = {}
        self.stats        = {"batches": 0, "frames_in": 0, "frames_out": 0}
        self._thread      = None
        self.capture_thread = None

    def start(self) -> None:

        if self._thread is None:
            self._thread = threading.Thread(target=self._work, daemon=True)
            self._thread.start()
        if self.capture_thread is None:
            self.capture_thread = threading.Thread(target=self._capture, daemon=True)
            self.capture_thread.start()

    def stop(self) -> None:
        self.stop.set()

    def push(self, frame: np.ndarray) -> None:
        stamp = time.time()

        with self._lock:
            self.preview = frame.copy()
            self.stats["frames_in"] += 1
        self.incoming.put((stamp, frame))

    def _capture(self) -> None:
        source = self.settings.camera_source

        try:
            if source.isdigit():
                cap = cv2.VideoCapture(int(source))
            else:
                cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        except Exception as exc:
            self.last_error = f"camera open error: {exc}"
            print(f"[camera] {self.last_error}")

            return

        if not cap.isOpened():
            self.last_error = f"camera open failed: {source}"
            print(f"[camera] {self.last_error}")

            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.settings.frame_width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.settings.frame_height))
        cap.set(cv2.CAP_PROP_FPS, float(self.settings.capture_fps))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        interval = 1.0 / max(1.0, self.settings.capture_fps)
        next_at  = time.perf_counter()
        self.camera_info = {
            "source": source,
            "opened": True,
            "fps": self.settings.capture_fps,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        print(f"[camera] {source} opened at {self.settings.capture_fps:g} fps")

        try:

            while not self.stop.is_set():
                ok, frame = cap.read()

                if ok and frame is not None:
                    self.push(frame)
                else:
                    self.last_error = "camera read failed"
                    time.sleep(interval)

                    continue
                next_at += interval
                delay = next_at - time.perf_counter()

                if delay > 0:
                    time.sleep(delay)
                else:
                    next_at = time.perf_counter()
        finally:
            cap.release()
            self.camera_info = {"source": source, "opened": False}

    def describe(self) -> dict:
        return {
            "person": self.person.describe(),
            "face": self.face.describe() if self.face is not None else None,
            "age_gender": self.age.describe() if self.age is not None else None,
        }

    def snapshot(self) -> dict:

        with self._lock:
            return {
                "result": {str(key): value for key, value in self.result.items()},
                "stats": dict(self.stats),
                "camera": dict(self.camera_info),
                "timings_ms": {key: round(value * 1000, 1) for key, value in self.timings.items()},
                "last_error": self.last_error,
            }

    def frames(self):
        interval = 1.0 / max(1.0, self.settings.stream_fps)

        while True:
            started = time.time()

            with self._lock:
                has_frame = self.preview is not None
                frame = self.preview.copy() if has_frame else _placeholder()
                people = [dict(person) for person in self.overlay_people]
                faces  = [dict(face) for face in self.overlay_faces]
                result = dict(self.result)
            if has_frame:
                self._draw(frame, people, faces, result)
            ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.settings.jpeg_quality])

            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
            time.sleep(max(0.0, interval - (time.time() - started)))

    def _work(self) -> None:
        batch   = []
        current = None
        deadline = 0.0

        while not self.stop.is_set():

            try:
                stamp, frame = self.incoming.get(timeout=0.05)
            except queue.Empty:

                if batch and time.time() >= deadline and self.incoming.qsize() == 0:
                    self._flush(batch)
                    batch, current, deadline = [], None, 0.0

                continue
            second = int(stamp)

            if current is None:
                current  = second
                deadline = second + 1.0

            if second != current:

                if batch:
                    self._flush(batch)
                batch    = []
                current  = second
                deadline = second + 1.0
            batch.append((stamp, frame))

            if time.time() >= deadline and self.incoming.qsize() == 0:
                self._flush(batch)
                batch, current, deadline = [], None, 0.0

        if batch:
            self._flush(batch)

    def _flush(self, batch: list) -> None:
        started = time.perf_counter()

        try:
            result = self._process(batch)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            print(f"[batch] {self.last_error}")

            return

        elapsed = round((time.perf_counter() - started) * 1000, 1)

        with self._lock:
            self.result          = result
            self.last_error      = None
            self.stats["batches"] += 1
            self.stats["frames_out"] += len(batch)
        faces    = sum(int(value.get("samples", 0)) for value in result.values())
        model_ms = sum(value * 1000 for value in self.timings.values())
        other_ms = max(0.0, elapsed - model_ms)
        print(
            f"[batch] in={len(batch)} top={self.top_frame} faces={faces} people={len(result)} ms={elapsed} "
            f"yolo={self.timings['yolo'] * 1000:.0f} "
            f"face={self.timings['face'] * 1000:.0f} "
            f"age={self.timings['age'] * 1000:.0f} "
            f"other={other_ms:.0f}"
        )

    def _process(self, batch: list) -> dict:
        with self._lock:
            previous = dict(self.result)

        timings     = {"yolo": 0.0, "face": 0.0, "age": 0.0}
        by_person   = defaultdict(list)
        best_index  = None
        best_score  = (-1.0, -1.0)
        best_view   = None
        best_people = []
        faces       = []

        for index, (_, frame) in enumerate(batch):
            started = time.perf_counter()
            people  = self._people(frame)
            timings["yolo"] += time.perf_counter() - started

            with self._lock:
                self.overlay_people = [dict(person) for person in people]

            if not people:

                continue
            score = (
                sum(float(person.get("conf", 0.0)) for person in people),
                max(sharpness(frame, person["box"]) for person in people),
            )

            if score > best_score:
                best_score  = score
                best_index  = index
                best_view   = frame
                best_people = people
        self.top_frame = 0 if best_index is None else best_index + 1

        if best_index is not None:
            started = time.perf_counter()
            faces = self._faces(best_view, best_people)
            timings["face"] += time.perf_counter() - started

            for face in faces:
                person_id = face.get("person_id")

                if person_id is not None:
                    by_person[person_id].append((face, best_view))

        result = {}

        for person_id, entries in by_person.items():
            values = []

            for face, view in entries:
                started = time.perf_counter()
                ok = self._attributes(view, face)
                timings["age"] += time.perf_counter() - started

                if not ok:

                    continue
                values.append((
                    face.get("age", 0),
                    face.get("gender", "Unknown"),
                    bool(face.get("looking")),
                    float(face.get("gconf", 0.0)),
                ))

            if values:
                result[person_id] = {
                    "age": _mode([value[0] for value in values]),
                    "gender": _mode([value[1] for value in values]),
                    "looking": _mode([value[2] for value in values]),
                    "samples": len(entries),
                    "confidence": round(sum(value[3] for value in values) / len(values), 3),
                    "model_runs": len(values),
                }
            else:
                prior = previous.get(person_id, {})
                result[person_id] = {
                    "age": prior.get("age", 0),
                    "gender": prior.get("gender", "Unknown"),
                    "looking": bool(prior.get("looking", False)),
                    "samples": len(entries),
                    "confidence": float(prior.get("confidence", 0.0)),
                    "model_runs": 0,
                }

            aggregated = result[person_id]

            for face, _ in entries:
                face["age"]         = aggregated["age"]
                face["gender"]      = aggregated["gender"]
                face["gconf"]       = aggregated["confidence"]
                face["looking"]     = aggregated["looking"]
                face["aggregated"]  = True

        with self._lock:
            self.overlay_faces = [dict(face) for face in faces]
        self.timings = timings

        return result

    def _people(self, frame: np.ndarray) -> list[dict]:
        boxes = []

        for detection in self.person.detect(frame):

            if detection.get("kind") != "human":

                continue
            confidence = float(detection.get("conf", 0.0))

            if confidence < self.settings.person_conf:

                continue
            box = [int(value) for value in detection["box"]]

            if box[2] - box[0] < 1 or box[3] - box[1] < 1:

                continue
            if box[3] - box[1] < self.settings.person_min_h:

                continue
            boxes.append((box, confidence))

        matched = self.tracker.match([box for box, _ in boxes])
        people  = []

        for (box, confidence), (person_id, _) in zip(boxes, matched):
            people.append({"box": box, "conf": confidence, "person_id": person_id})

        return people

    def _faces(self, frame: np.ndarray, people: list[dict]) -> list[dict]:

        if self.face is None:

            return []
        height, width = frame.shape[:2]
        found         = []

        for person in people:
            x1, y1, x2, y2 = person["box"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            if x2 - x1 < 2 or y2 - y1 < 2:

                continue

            try:
                detected = self.face.detect(frame[y1:y2, x1:x2])
            except Exception as exc:
                print(f"[face] {exc}")

                continue

            for face in detected:
                if float(face.get("conf", 0.0)) < self.settings.face_score:

                    continue
                fx1, fy1, fx2, fy2 = (int(value) for value in face["box"])
                face["box"] = [fx1 + x1, fy1 + y1, fx2 + x1, fy2 + y1]
                landmarks   = face.get("landmarks")

                if landmarks:
                    face["landmarks"] = [[int(px) + x1, int(py) + y1] for px, py in landmarks]
                face["person_id"] = person["person_id"]
                face["human_box"] = person["box"]
                found.append(face)

        unique = []

        for face in found:

            if any(iou(face["box"], other["box"]) > 0.7 for other in unique):

                continue
            unique.append(face)

        for face in unique:
            self._prepare_frame_face(frame, face)

        return unique

    def _prepare_frame_face(self, frame: np.ndarray, face: dict) -> None:
        face.setdefault("gender", "Unknown")
        face.setdefault("age", 0)
        face.setdefault("gconf", 0.0)
        x1, y1, x2, y2 = face["box"]
        face["sharpness"] = round(sharpness(frame, face["box"]), 1)
        face["area"]      = (x2 - x1) * (y2 - y1)
        face["eligible"]  = True

    def _attributes(self, frame: np.ndarray, face: dict) -> bool:

        if self.age is None:

            return False
        sample = self.age.estimate(frame, face, self.settings.age_align)

        if sample is None:

            return False
        raw_age         = float(sample.raw_age)
        face["age"]     = int(round(raw_age))
        face["raw_age"] = round(raw_age, 1)
        logits          = np.asarray(sample.gender_logits, dtype=np.float32)
        logits          = logits - float(np.max(logits))
        probs           = np.exp(logits)
        probs          /= max(float(np.sum(probs)), 1e-12)
        chosen          = int(np.argmax(probs))
        face["gender"]  = "Male" if chosen == 1 else "Female"
        face["gconf"]   = round(float(probs[chosen]), 3)
        face["align"]   = sample.align

        return True

    def _draw(self, frame: np.ndarray, people: list[dict], faces: list[dict], result: dict) -> None:
        image = Image.fromarray(frame)
        draw  = ImageDraw.Draw(image)
        font  = _font(18)

        for face in faces:
            x1, y1, x2, y2 = face["box"]
            draw.rectangle((x1, y1, x2, y2), outline=(255, 200, 0), width=1)

        for person in people:
            person_id         = person["person_id"]
            x1, y1, x2, y2    = person["box"]
            aggregated        = result.get(person_id, {})
            gender            = aggregated.get("gender", "Unknown")
            age               = aggregated.get("age", 0)
            text              = f"#{person_id} {gender} {age}y {person['conf']:.2f}"
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            width, height     = right - left, bottom - top
            text_y            = max(0, y1 - height - 6)
            draw.rectangle((x1, text_y, x1 + width + 6, text_y + height + 4), fill=(0, 0, 0))
            draw.text((x1 + 3, text_y + 2), text, font=font, fill=(255, 255, 255))
            draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=2)

        frame[:] = np.asarray(image)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.from_env()

    try:
        cv2.setNumThreads(max(1, min(4, settings.cv_threads)))
    except Exception:
        pass

    pipeline = Pipeline(settings)
    pipeline.start()
    app.state.runtime = SimpleNamespace(settings=settings, pipeline=pipeline)

    try:
        yield
    finally:
        pipeline.stop()


app = FastAPI(title="AI Camera - Python capture + per-person mode", lifespan=lifespan)


@app.get("/ubuntu-sans.ttf")
def ubuntu_sans():

    return FileResponse(FONT_PATH, media_type="font/ttf")


@app.get("/", dependencies=[Depends(require_token)])
def index():

    return FileResponse(INDEX_PATH, media_type="text/html")


@app.get("/api/output/stream")
def output_stream(request: Request, _=Depends(require_token)):

    return StreamingResponse(
        request.app.state.runtime.pipeline.frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/result", dependencies=[Depends(require_token)])
def result(request: Request):

    return request.app.state.runtime.pipeline.snapshot()


@app.get("/api/health", dependencies=[Depends(require_token)])
def health(request: Request):
    pipeline = request.app.state.runtime.pipeline
    details  = pipeline.describe()
    age      = details.get("age_gender") or {}

    return {
        "age_gender_backend": age.get("backend"),
        "age_gender_model_kind": age.get("model_kind"),
        "age_source": age.get("age_source"),
        "age_loaded": bool(age.get("age_loaded")),
        "camera": dict(pipeline.camera_info),
        "models": details,
    }


@app.post("/api/debug/frame", dependencies=[Depends(require_token)])
async def debug_frame(request: Request):
    data  = await request.body()
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)

    if frame is None:

        raise HTTPException(status_code=400, detail="Invalid JPEG bytes")
    request.app.state.runtime.pipeline.push(frame)

    return {"queued": 1}


@app.post("/api/debug/synthetic", dependencies=[Depends(require_token)])
def debug_synthetic(request: Request):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (120, 80), (420, 440), (60, 60, 60), -1)
    cv2.putText(frame, "DEBUG FRAME", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2)
    request.app.state.runtime.pipeline.push(frame)

    return {"queued": 1}
