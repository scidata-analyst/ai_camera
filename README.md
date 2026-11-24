# AI Camera — RK3576 + YOLO11n + RetinaFace + Age/Gender

Camera stream-er pipeline:

- YOLO11n detects `person` and COCO objects.
- RetinaFace detects face boxes and five landmarks.
- MiVOLO v2 (face-only) estimates gender and age from each aligned face crop.
- FairFace and the dlib age predictor remain available as an alternative path.
- WHENet estimates head pose / whether a face is looking at the camera.

RetinaFace is the face detector. By default, MiVOLO v2 supplies both age and gender through RKNN on the NPU; FairFace + dlib age stay in the image and can be selected again with `AGE_GENDER_RKNN_MODEL`.

## Code structure

The pipeline follows a domain-driven layout:

| Folder | Responsibility |
| --- | --- |
| `app/config/` | Environment settings and shared constants |
| `app/domain/detection/` | Human detection port and filtering service |
| `app/domain/face/` | Face detection service and blur/size quality gate |
| `app/domain/age/` | Raw age decoding and calibration |
| `app/domain/gender/` | Gender softmax/argmax decoding |
| `app/domain/attributes/` | Shared age/gender estimator port |
| `app/domain/aggregation/` | Mean/median/mode window and per-person attribute smoothing |
| `app/domain/headpose/` | Head-pose analyzer port |
| `app/application/` | Frame processor and cross-domain orchestration |
| `app/infrastructure/rknn/` | RKNN adapters for YOLO, RetinaFace, MiVOLO, FairFace, dlib age and WHENet |
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

Model binaries are stored in Git LFS. After cloning, run
`git lfs install && git lfs pull` before building the Docker image; otherwise
the working tree contains small pointer files instead of the real models.

The ONNX models are in `models/`:

- `RetinaFace_mobile320.onnx` — official RKNN Model Zoo RetinaFace mobile model
- `mivolo-face.onnx` — MiVOLO v2 face-only age/gender model, with `mivolo-face.json` metadata
- `fairface.onnx` — FairFace ResNet-34 race/gender/age model (MIT source, CC BY 4.0 weights)
- `dlib-age-v1.onnx` — dlib ResNet-10 age predictor, 81 age classes (CC0-1.0, contributed by Cydral Technology)
- Existing YOLO11n and WHENet files

On an x86_64 Linux PC with RKNN-Toolkit2 installed, convert RetinaFace, MiVOLO, FairFace and the dlib age predictor for RK3576 manually:

```bash
python tools/export_face_models_rknn.py --target rk3576
```

This creates:

- `models/RetinaFace_mobile320-rk3576.rknn`
- `models/mivolo-face-rk3576.rknn`
- `models/fairface-rk3576.rknn`
- `models/dlib-age-rk3576.rknn`

MiVOLO receives a 224×224 RGB letterbox crop of the aligned face; normalization is compiled into the RKNN model, so runtime input is `uint8` NHWC with no extra scaling. It returns male logit, female logit and a normalized age; age is decoded as `value * 94 + 48` and clamped to 1–95. The alternative FairFace path uses the same aligned crop, with dlib age on a separate 64×64 model; FairFace's nine bin estimate is only a fallback if dlib age fails.

The app is **NPU-only**: YOLO, RetinaFace, MiVOLO, FairFace, the dlib age predictor and WHENet all run through RKNNLite. CPU/ONNX fallbacks were removed — a missing or failed RKNN model makes startup fail with a clear error instead of silently using CPU. Compose only mounts `models/`; it does not start conversion.

For commercial deployment, check the pretrained model licenses first: MiVOLO is Apache-2.0 at its source, the FairFace weights are CC BY 4.0 and the dlib age weights are CC0-1.0. See the [MiVOLO project](https://github.com/WildChlamydia/MiVOLO), the [FairFace project](https://github.com/dchen236/FairFace), the [dlib age models](https://github.com/davisking/dlib-models) and the [RKNN Model Zoo RetinaFace example](https://github.com/airockchip/rknn_model_zoo/tree/main/examples/RetinaFace).

## Main settings

| Variable | Default | Purpose |
| --- | --- | --- |
| `YOLO_BACKEND` | `rknn` | YOLO inference on NPU only; `cpu`/`auto` rejected |
| `IMG_SIZE` | `320` | YOLO input size |
| `FACE_BACKEND` | `rknn` | RetinaFace on NPU only; `off` disables face stage |
| `FACE_MODEL` | `/app/models/RetinaFace_mobile320.onnx` | Legacy path (unused in NPU-only mode) |
| `FACE_RKNN_MODEL` | `/app/models/RetinaFace_mobile320-rk3576.rknn` | RetinaFace RKNN model |
| `AGE_GENDER_BACKEND` | `rknn` | Age/gender estimator on NPU only; `off` disables it |
| `AGE_GENDER_MODEL` | `/app/models/mivolo-face.onnx` | Legacy path (unused in NPU-only mode) |
| `AGE_GENDER_RKNN_MODEL` | `/app/models/mivolo-face-rk3576.rknn` | MiVOLO RKNN model; set to `fairface-rk3576.rknn` to use FairFace + dlib age |
| `AGE_RKNN_MODEL` | `/app/models/dlib-age-rk3576.rknn` | dlib age RKNN model; used only with the FairFace path, needs restart |
| `CONF_THRESHOLD` | `0.5` | NMS score floor for person boxes |
| `PERSON_CONF` | `0.65` | Minimum score required for a `person` box |
| `PERSON_MIN_H` | `0` | Minimum person-box height in pixels; `0` = no size limit |
| `PERSON_REQUIRE_FACE` | `0` | Set to `1` to accept a person box only when a face lies inside it |
| `DET_STALE_SEC` | `0.2` | Max age of a reused box before YOLO is forced to re-run |
| `YOLO_DEBUG` | `0` | Set to `1` to print the max NPU person score once at startup |
| `FACE_SCORE` | `0.6` | Minimum RetinaFace confidence (Compose: `0.65`) |
| `FACE_MIN_SIZE` | `60` | Minimum face side in pixels for age/gender; `0` = analyze every face |
| `FACE_BLUR` | `80` | Min Laplacian variance of the face crop; `0` disables the blur gate. Blurry frames get no new age/gender prediction |
| `AGE_GENDER_ALIGN` | `arcface` | Face crop mode for age/gender: `arcface` (5-point alignment, default), `roll` (eye-line levelled), `box` (plain crop) |
| `FACE_EVERY` | `1` | Run face detection every N inference frames; every detected person is analyzed |
| `INTERVAL` | `1` | Single interval in seconds for counts and age/gender. Samples are collected for this long, results publish once, then all previous samples are cleared. Change live from the Setting popup |
| `HEADPOSE_ENABLE` | `1` | WHENet head-pose stage on NPU |
| `HEADPOSE_BACKEND` | `rknn` | Head pose on NPU only; `off` disables it |
| `HEADPOSE_RKNN_MODEL` | `/app/models/whenet-rk3576.rknn` | WHENet RKNN model |

Age and gender are visual estimates; small, blurry, profile, or poorly lit faces can reduce reliability. They do not identify a person's actual gender or exact age. By default both come from MiVOLO's male/female logits and normalized age output. If `AGE_GENDER_RKNN_MODEL` is pointed at `fairface-rk3576.rknn`, gender comes from FairFace and age from the dlib 81-class predictor, with FairFace's nine age bins only as a fallback. Predictions are collected for `INTERVAL` seconds (1 by default, changeable live from the Setting popup), then one result is published per person and the collected samples are cleared. Tracks are matched by **human-box IoU**, so multiple people keep separate results. The displayed age and gender are majority votes of that person's samples in the interval (for example, 6 male vs 4 female gives male). A tie uses the most recent prediction. Between two publish moments the last valid age/gender is held, so a failed or skipped frame never hides the label. Text labels are drawn only on human boxes; face boxes keep rectangles/landmarks without text.

The page is served from `app/interfaces/controllers/index.html`. Ubuntu Sans is bundled at `app/interfaces/controllers/UbuntuSans.ttf` and served at `/ubuntu-sans.ttf`, so both the page and the video overlay use the same font without network access. The **Setting** popup controls every supported environment variable. Fields marked `needs restart` are saved but require a container restart; all other fields apply immediately. Saved values are written to `SETTINGS_FILE` (default `app/config/saved_settings.json`; Compose mounts `./settings` at `/app/config`), so they survive restarts. **Reset All** restores the startup values and removes the saved file. If `API_TOKEN` is set, open `/?token=<token>` so the page can authenticate its API requests.

Quality gates: faces smaller than `FACE_MIN_SIZE` or crops with Laplacian variance below `FACE_BLUR` are skipped, so a blurry frame does not add a bad sample. `AGE_GENDER_ALIGN=arcface` uses the five RetinaFace landmarks for full alignment; `roll` only levels the eye line and `box` uses a plain crop. MiVOLO receives the aligned crop as a 224×224 RGB letterbox; FairFace and dlib use their own fixed crops. `/api/faces` reports `sharpness`, `blurry`, `raw_age` and `align` for tuning.

## Existing exporters

YOLO and WHENet model export tools remain manual. For example:

```bash
python tools/export_rknn.py --imgsz 320
python tools/export_onnx_rknn.py --model models/whenet.onnx \
  --out models/whenet-rk3576.rknn
```

## Hardware

The Compose service targets the RK3576 board, maps `/dev/dri/renderD129`, and mounts `./models:/app/models`. Check `/api/health` to confirm each backend is `rknn`; a missing/failed RKNN model now fails startup instead of falling back to CPU.
