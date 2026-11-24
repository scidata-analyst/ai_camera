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
| `FACE_SCORE` | `0.6` | Minimum RetinaFace confidence |
| `FACE_MIN_SIZE` | `60` | Smaller face boxes show gender as `Unknown` |
| `FACE_EVERY` | `1` | Run face detection every N inference frames |
| `MAX_FACES` | `0` | Per-frame face cap; `0` = unlimited (detect all) |
| `HEADPOSE_ENABLE` | `1` | WHENet head-pose stage on NPU |
| `HEADPOSE_BACKEND` | `rknn` | Head pose on NPU only; `off` disables it |
| `HEADPOSE_RKNN_MODEL` | `/app/models/whenet-rk3576.rknn` | WHENet RKNN model |

GenderAge is a visual estimate; small, blurry, profile, or poorly lit faces can reduce reliability. It does not identify a person's actual gender or exact age. Displayed age/gender labels average every detected human frame for 1 second (gender = average signed confidence, age = rounded mean), then **lock** that value for the tracked face — the label neither flickers nor changes without a new/moving face. Tracks are matched per person by face-box IoU, so multiple people in one frame keep separate ages/genders; a missing face keeps its own last value and never copies another person's.

## Existing exporters

YOLO and WHENet model export tools remain manual. For example:

```bash
python tools/export_rknn.py --imgsz 320
python tools/export_onnx_rknn.py --model models/whenet.onnx \
  --out models/whenet-rk3576.rknn
```

## Hardware

The Compose service targets the RK3576 board, maps `/dev/dri/renderD129`, and mounts `./models:/app/models`. Check `/api/health` to confirm each backend is `rknn`; a missing/failed RKNN model now fails startup instead of falling back to CPU.
