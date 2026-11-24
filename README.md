# AI Camera — RK3576 + YOLO11n + RetinaFace + Age/Gender

Camera stream-er pipeline:

- YOLO11n detects `person` and COCO objects.
- RetinaFace detects face boxes and five landmarks.
- InsightFace GenderAge estimates gender and age from each face crop.
- WHENet estimates head pose / whether a face is looking at the camera.

RetinaFace is the face detector. Age and gender come from the separate, lightweight `genderage.onnx` head. MiVOLO is removed.

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
| `PERSON_CONF` | `0.65` | Separate, higher score required for a `person` box (false "Human" fix) |
| `PERSON_MIN_H` | `0` | Minimum person-box height in pixels; `0` = no size limit |
| `PERSON_REQUIRE_FACE` | `0` | Set to `1` to accept a person box only when a face lies inside it |
| `DET_STALE_SEC` | `0.2` | Max age of a reused box before YOLO is forced to re-run |
| `YOLO_DEBUG` | `0` | Set to `1` to print the max NPU person score once at startup |
| `FACE_SCORE` | `0.6` | Minimum RetinaFace confidence (Compose: `0.65`) |
| `FACE_MIN_SIZE` | `60` | Minimum face side in pixels for age/gender; `0` = analyze every face |
| `FACE_BLUR` | `80` | Min Laplacian variance of the face crop; `0` disables the blur gate. Blurry frames get no new age/gender prediction |
| `AGE_GENDER_ALIGN` | `roll` | Face crop mode for age/gender: `box` (official), `roll` (eye-line levelled, default), `arcface` (full 5-point alignment, experimental) |
| `AGE_SCALE` | `1.0` | Age calibration multiplier: `age = round(raw_age * AGE_SCALE + AGE_BIAS)` |
| `AGE_BIAS` | `0.0` | Age calibration offset in years |
| `FACE_EVERY` | `1` | Run face detection every N inference frames |
| `MAX_FACES` | `0` | Per-frame face cap; `0` = unlimited (detect all) |
| `HEADPOSE_ENABLE` | `1` | WHENet head-pose stage on NPU |
| `HEADPOSE_BACKEND` | `rknn` | Head pose on NPU only; `off` disables it |
| `HEADPOSE_RKNN_MODEL` | `/app/models/whenet-rk3576.rknn` | WHENet RKNN model |

GenderAge is a visual estimate; small, blurry, profile, or poorly lit faces can reduce reliability. It does not identify a person's actual gender or exact age. Displayed age/gender labels average every detected human frame for 1 second (gender = average signed confidence, age = rounded mean), then **lock** that value for the tracked face to avoid flicker. The lock is not permanent: a track expires after a few seconds without the face, and a run of consistent fresh predictions that disagree with the locked gender/age breaks the lock and re-locks with the new value. Tracks are matched per person by face-box IoU, so multiple people in one frame keep separate ages/genders; a missing face never copies another person's value. Text labels are drawn only on human boxes; face boxes keep rectangles/landmarks without text.

Quality gates: faces smaller than `FACE_MIN_SIZE` or crops with Laplacian variance below `FACE_BLUR` are skipped, so a blurry frame cannot create a wrong locked value. `AGE_GENDER_ALIGN=roll` keeps the model's original 1.5x framing but levels the eye line from the five RetinaFace landmarks; `arcface` does full landmark alignment (experimental — this model was trained on box-centre crops, so A/B test before using it). `/api/faces` reports `sharpness`, `blurry`, `raw_age` and `align` for tuning. For systematic age bias, collect a few `(raw_age, real_age)` pairs and set `AGE_SCALE`/`AGE_BIAS` from a least-squares fit (`AGE_SCALE = cov(raw, real) / var(raw)`, `AGE_BIAS = mean(real) - AGE_SCALE * mean(raw)`); leave both at defaults until you have samples.

## Existing exporters

YOLO and WHENet model export tools remain manual. For example:

```bash
python tools/export_rknn.py --imgsz 320
python tools/export_onnx_rknn.py --model models/whenet.onnx \
  --out models/whenet-rk3576.rknn
```

## Hardware

The Compose service targets the RK3576 board, maps `/dev/dri/renderD129`, and mounts `./models:/app/models`. Check `/api/health` to confirm each backend is `rknn`; a missing/failed RKNN model now fails startup instead of falling back to CPU.
