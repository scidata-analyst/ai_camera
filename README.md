# AI Camera — RK3576 + YOLO11n + RetinaFace + Age/Gender

Camera stream-er pipeline:

- YOLO11n detects `person` and COCO objects.
- RetinaFace detects face boxes and five landmarks.
- InsightFace GenderAge estimates gender and age from each face crop.
- WHENet estimates head pose / whether a face is looking at the camera.

RetinaFace is the face detector. Age and gender come from the separate, lightweight `genderage.onnx` head. MiVOLO is removed.

## Code structure

The pipeline follows a domain-driven layout:

| Folder | Responsibility |
| --- | --- |
| `app/config/` | Environment settings and shared constants |
| `app/domain/detection/` | Human detection port and filtering service |
| `app/domain/face/` | Face detection service and blur/size quality gate |
| `app/domain/age/` | Raw age decoding and `AGE_SCALE`/`AGE_BIAS` calibration |
| `app/domain/gender/` | Gender softmax/argmax decoding |
| `app/domain/attributes/` | Shared age/gender estimator port |
| `app/domain/aggregation/` | Mean/median/mode window and per-person attribute smoothing |
| `app/domain/headpose/` | Head-pose analyzer port |
| `app/application/` | Frame processor and cross-domain orchestration |
| `app/infrastructure/rknn/` | RKNN adapters for YOLO, RetinaFace, GenderAge and WHENet |
| `app/infrastructure/camera/` | OpenCV frame source |
| `app/infrastructure/rendering/` | OpenCV drawing/overlay renderer |
| `app/interfaces/` | FastAPI controllers, presenters and flip state |
| `app/main.py` | Composition root: wires settings, adapters, services and routers |

All HTTP endpoints and JSON payload keys are unchanged; only the internal module layout was reorganized.

## Run

```bash
docker compose up --build -d
```

Open `http://localhost:8080`. Status endpoints:

- `/api/health` — active model backends
- `/api/stats` — per-stage timing and age/gender inference count
- `/api/faces` — face boxes, landmarks, age and gender

`docker compose build` and `docker compose up` do not export models. RKNN conversion is a separate manual command below.

## Models

The ONNX models are in `models/`:

- `RetinaFace_mobile320.onnx` — official RKNN Model Zoo RetinaFace mobile model
- `genderage.onnx` — InsightFace GenderAge attribute model
- Existing YOLO11n and WHENet files

On an x86_64 Linux PC with RKNN-Toolkit2 installed, convert RetinaFace and GenderAge for RK3576 manually:

```bash
python tools/export_face_models_rknn.py --target rk3576
```

This creates:

- `models/RetinaFace_mobile320-rk3576.rknn`
- `models/genderage-rk3576.rknn`

The app is **NPU-only**: YOLO, RetinaFace, GenderAge and WHENet all run through RKNNLite. CPU/ONNX fallbacks were removed — a missing or failed RKNN model makes startup fail with a clear error instead of silently using CPU. Compose only mounts `models/`; it does not start conversion.

For commercial deployment, check the pretrained model licenses first: InsightFace states its pretrained models are for non-commercial research use. See the [InsightFace model zoo](https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md) and the [RKNN Model Zoo RetinaFace example](https://github.com/airockchip/rknn_model_zoo/tree/main/examples/RetinaFace).

## Main settings

| Variable | Default | Purpose |
| --- | --- | --- |
| `YOLO_BACKEND` | `rknn` | YOLO inference on NPU only; `cpu`/`auto` rejected |
| `IMG_SIZE` | `320` | YOLO input size |
| `FACE_BACKEND` | `rknn` | RetinaFace on NPU only; `off` disables face stage |
| `FACE_MODEL` | `/app/models/RetinaFace_mobile320.onnx` | Legacy path (unused in NPU-only mode) |
| `FACE_RKNN_MODEL` | `/app/models/RetinaFace_mobile320-rk3576.rknn` | RetinaFace RKNN model |
| `AGE_GENDER_BACKEND` | `rknn` | GenderAge on NPU only; `off` disables it |
| `AGE_GENDER_MODEL` | `/app/models/genderage.onnx` | Legacy path (unused in NPU-only mode) |
| `AGE_GENDER_RKNN_MODEL` | `/app/models/genderage-rk3576.rknn` | GenderAge RKNN model |
| `CONF_THRESHOLD` | `0.5` | NMS score floor for person boxes |
| `PERSON_CONF` | `0.65` | Minimum score required for a `person` box |
| `PERSON_MIN_H` | `0` | Minimum person-box height in pixels; `0` = no size limit |
| `PERSON_REQUIRE_FACE` | `0` | Set to `1` to accept a person box only when a face lies inside it |
| `DET_STALE_SEC` | `0.2` | Max age of a reused box before YOLO is forced to re-run |
| `YOLO_DEBUG` | `0` | Set to `1` to print the max NPU person score once at startup |
| `FACE_SCORE` | `0.6` | Minimum RetinaFace confidence (Compose: `0.65`) |
| `FACE_MIN_SIZE` | `60` | Minimum face side in pixels for age/gender; `0` = analyze every face |
| `FACE_BLUR` | `80` | Min Laplacian variance of the face crop; `0` disables the blur gate. Blurry frames get no new age/gender prediction |
| `AGE_GENDER_ALIGN` | `roll` | Face crop mode for age/gender: `box` (official), `roll` (eye-line levelled, default), `arcface` (full 5-point alignment, experimental) |
| `FACE_EVERY` | `1` | Run face detection every N inference frames; every detected person is analyzed |
| `SMOOTH_WINDOW` | `1` | Age/gender interval in seconds: samples are collected for this long, then labels update once. Change live from the Setting popup |
| `COUNT_WINDOW` | `1` | Rolling window in seconds for displayed counts; set `5` or `10` from the Setting popup |
| `HEADPOSE_ENABLE` | `1` | WHENet head-pose stage on NPU |
| `HEADPOSE_BACKEND` | `rknn` | Head pose on NPU only; `off` disables it |
| `HEADPOSE_RKNN_MODEL` | `/app/models/whenet-rk3576.rknn` | WHENet RKNN model |

GenderAge is a visual estimate; small, blurry, profile, or poorly lit faces can reduce reliability. It does not identify a person's actual gender or exact age. Predictions are collected for `SMOOTH_WINDOW` seconds (1 by default, changeable live from the Setting popup), then one result is published per person. Tracks are matched by **human-box IoU**, so multiple people keep separate results. The displayed age is the rounded mean of that person's age samples in the interval; the displayed gender is the majority vote of the interval's predictions (for example, 6 male vs 4 female gives male). A tie uses the most recent prediction. Between two publish moments the last valid age/gender is held, so a failed or skipped frame never hides the label. Text labels are drawn only on human boxes; face boxes keep rectangles/landmarks without text.

The page is served from `app/interfaces/controllers/index.html`. The **Setting** popup controls every supported environment variable. Fields marked `needs restart` are saved but require a container restart; all other fields apply immediately. Saved values are written to `SETTINGS_FILE` (default `app/config/saved_settings.json`; Compose mounts `./settings` at `/app/config`), so they survive restarts. **Reset All** restores the startup values and removes the saved file. If `API_TOKEN` is set, open `/?token=<token>` so the page can authenticate its API requests.

Quality gates: faces smaller than `FACE_MIN_SIZE` or crops with Laplacian variance below `FACE_BLUR` are skipped, so a blurry frame does not add a bad sample. `AGE_GENDER_ALIGN=roll` keeps the model's original 1.5x framing but levels the eye line from the five RetinaFace landmarks; `arcface` does full landmark alignment (experimental — this model was trained on box-centre crops, so A/B test before using it). `/api/faces` reports `sharpness`, `blurry`, `raw_age` and `align` for tuning.

## Existing exporters

YOLO and WHENet model export tools remain manual. For example:

```bash
python tools/export_rknn.py --imgsz 320
python tools/export_onnx_rknn.py --model models/whenet.onnx \
  --out models/whenet-rk3576.rknn
```

## Hardware

The Compose service targets the RK3576 board, maps `/dev/dri/renderD129`, and mounts `./models:/app/models`. Check `/api/health` to confirm each backend is `rknn`; a missing/failed RKNN model now fails startup instead of falling back to CPU.
