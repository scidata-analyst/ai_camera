# AI Camera — Human + Face + Gender (RK3576 + Docker + YOLOv11n + YuNet + MiVOLO)

USB webcam (`/dev/video0`) theke frame niye YOLOv11n + YuNet + MiVOLO diye detect kore.
- `person` (class 0) -> **Human** (green box)
- onno sob class -> **Non-Human** (orange box)
- YuNet -> **Face** (cyan box + 5 landmark dot: eyes, nose, mouth)
- MiVOLO V2 (face + body dual input) -> **Male/Female + age** (face label e, e.g. `F0.93 27y`)
- Live MJPEG stream: `http://localhost:8080`

## Quick start (RK3576 / Linux)

```bash
# 1. build + run (camera soho)
docker compose up --build

# camera chara test (laptop e /dev/video0 na thakle)
# docker-compose.yml e camera device mapping comment kore:
docker compose up --build

# 2. browser e kholo
http://localhost:8080
http://localhost:8080/video_feed
http://localhost:8080/api/stats
http://localhost:8080/api/faces
http://localhost:8080/api/health
```

## Env config

| Var | Default | Note |
|-----|---------|------|
| CAMERA_SOURCE | 0 | `/dev/video0`, RTSP URL ba video file o cholbe |
| MODEL_NAME | /app/models/yolo11n.pt | YOLO weights ekhanei download hoy; export-o `models/` e hoy |
| YOLO_BACKEND | rknn | RKNN model/runtime na hole CPU fallback |
| RKNN_MODEL | /app/models/yolo11n-rk3576.rknn | YOLO RK3576 model |
| CONF_THRESHOLD | 0.5 | 0.3-0.7 |
| IMG_SIZE | 384 | 320=fast, 480=accurate, 384=balance (32 er multiple) |
| FRAME_WIDTH/HEIGHT | 640x480 | camera resolution |
| JPEG_QUALITY | 70 | stream quality |
| SMOOTH_WINDOW | 1.0 | 1 sec window te mean/median/mode, stable = mode |
| FACE_ENABLE | 1 | 0 korle face off |
| FACE_SCORE | 0.6 | YuNet conf threshold |
| FACE_MODEL | /app/models/yunet.onnx | auto-download hoy |
| FACE_BACKEND | rknn | `rknn/auto/cpu/opencl`; RKNN init fail hole OpenCV fallback |
| FACE_RKNN_MODEL | /app/models/yunet-rk3576.rknn | YuNet RK3576 model |
| FACE_MIN_SIZE | 60 | choto face e gender Unknown (vul prediction komate) |
| MIVOLO_ENABLE | 1 | 0 korle age/gender off |
| MIVOLO_MODEL | /app/models/mivolo_v2 | local MiVOLO checkpoint folder (~115MB) |
| MIVOLO_BACKEND | rknn | RKNN runtime fail hole CPU fallback |
| MIVOLO_RKNN_MODEL | /app/models/mivolo_v2_rk3576.rknn | converted MiVOLO model path |
| MIVOLO_REFRESH | 1.0 | proti sec e re-check -> sec proti median (barale CPU kombbe) |
| MIVOLO_BLUR | 80 | er niche sharpness hole age refresh off (motion blur fix) |
| FACE_EVERY | 2 | YuNet proti N infer e 1 bar, majhe box reuse (CPU ~30% save) |
| HEADPOSE_PROVIDER | auto | ORT provider list, auto = GPU try + CPU fallback |
| HEADPOSE_BACKEND | rknn | `rknn/auto/onnx`; RKNN init fail hole ONNX fallback |
| HEADPOSE_RKNN_MODEL | /app/models/whenet-rk3576.rknn | WHENet RK3576 model |
| OPENCV_OPENCL | auto | `auto/off` — race jit-lei OpenCL on |
| HEADPOSE_ENABLE | 1 | 0 korle looking off |
| HEADPOSE_MODEL | /app/models/whenet.onnx | auto-download (PINTO0309, 17MB) |
| LOOK_YAW / LOOK_PITCH | 25 / 20 | looking threshold degree |

## Speed design (lag-free)
- Inference background thread e chole, MJPEG stream alada ~25-30fps e serve hoy.
- Tai YOLO slow holeo video atke na — box update infer rate e (~2-8fps) hoy.
- `IMG_SIZE` komale box update fast, barale accuracy valo.

## Model files
- Download kora ebong converted model `models/` folder-e rakha hoy. App-er
  YOLO `.pt` download path `/app/models/yolo11n.pt`; YuNet/WHENet ONNX,
  MiVOLO checkpoint, ebong char-ti RKNN model-o ei mounted folder-e thake.
- Export scripts manual bhabe chalate hoy; `docker compose build` ba `up`
  konotai model export/conversion start kore na. Compose sudhu age theke thaka
  `models/` artifacts mount kore app chalaye.
- MiVOLO checkpoint ekhane `models/mivolo_v2/`; RK3576 conversion ekhane
  `models/mivolo_v2_rk3576.rknn`. YOLO conversion ekhane
  `models/yolo11n-rk3576.rknn`.

## Looking at camera (WHENet head pose)
- Per face: `yaw/pitch/roll` degree + `looking` bool.
- `|yaw|<=25° and |pitch|<=20°` → **Looking** (green box + 👀 label).
- `/api/faces` e full JSON; header e `Look:x` count. CPU te ~15ms/face.

## Gender + Age (MiVOLO V2)
- Per face: `gender Male/Female + gconf` + `age` (bochor). Adience gender 97.3.
- Input dual: YuNet face crop + YOLO human box (body); body na pele face-only.
- Video label: `M0.91 27y` / `F0.93 24y`; `/api/faces` e full JSON; header e `M:x W:y` count.
- RK3576 e `mivolo_v2_rk3576.rknn` thakle MiVOLO RKNN NPU-te chole;
  file na thakle ba runtime unavailable hole PyTorch CPU fallback hoy.
- Face cache ache: ekoi face e
  proti sec e 1 bar re-check hoy, majher frame e cache theke (0ms).
  1-sec window er readings er median/majority = oi sec er result.
- **Movement fix:** blurry frame (motion blur) ba side-face e age refresh hoy na —
  ager valoi thake; proti 1-sec window er readings er **median age + majority gender**
  publish hoy (window khali thakle ager stable value thake).
- Bachcha/side-face e age ±5y hote pare.

## Face (YuNet) result
- Per face: `box [x1,y1,x2,y2]` + `conf` + `landmarks[5]` (eyes, nose, mouth corners).
- `/api/faces` e JSON, video te cyan box + red dot.
- Smoothing face er jonneo 1-sec mean/median/mode hoy.

## Smoothing (1 sec update)
- `human/non_human/face/male/female` count 1-sec window e joma hoy.
- Proti 1 sec por por `mean / median / mode` publish hoy, UI + `/api/stats` e dekha jay.
- Choto face (<FACE_MIN_SIZE) e gender `Unknown` dekhay, count eo dhore na — vul M/F komate.
- Web UI panel o 1 sec por por auto-refresh hoy (stable M/W soho).

## NPU / GPU (RK3576, renderD129)
- `docker-compose.yml` e `/dev/dri:/dev/dri` mount ache. Ei Ubuntu/Rockchip
  setup-e RKNPU `/dev/dri/renderD129` hisebe expose hoy; eta `/dev/rknpu` node
  na-o hote pare. Compose-e ei node ebong `/proc/device-tree/compatible`
  RKNN Lite-er jonno map kora ache; `/sys` read-only mount ebong privileged
  access runtime-er container check-er jonno deoa ache.
- NPU (RKNN): `models/` folder-e YOLO, YuNet, MiVOLO, WHENet-er RKNN files
  thakte hobe. Compose-e char model-er backend `rknn` set ache; NPU/runtime
  init fail hole app fallback backend ney, `/api/health` e active backend dekho.
  `/dev/dri:/dev/dri` mount NPU render node container-e dey; `.rknn` file ba
  NPU runtime/device init na hole app CPU fallback kore. ARM64 image build-e
  `rknn-toolkit-lite2` ebong matching version-er `librknnrt.so` install hoy.
  Model convert korte:
  Export (PC te 1 bar):
  ```bash
  pip install rknn-toolkit2 ultralytics onnx
  python tools/export_rknn.py --imgsz 320
  # int8 (fast, calib image list lagbe):
  # python tools/export_rknn.py --quant i8 --dataset calib.txt --imgsz 320
  ```
  Export seshe `models/yolo11n-rk3576.rknn` board-er project-er `models/`
  folder-e copy kore `docker compose up --build -d` chalao. `/api/health` e
  `backend: "rknn"` ebong `npu_active: true` hole YOLO NPU-te cholche.
  `/api/health` e `backend`, `face_backend`, `mivolo_backend`,
  `headpose_backend` sob `rknn` hole char-ti neural network-i NPU-te active.
  `./models` volume-er madhyome `.rknn` files container-e mount hoy. PC theke
  RK3576 board-e `models/` folder copy kore `docker compose up --build -d`
  chalao. Exporter chalate:
  ```bash
  python tools/export_mivolo_rknn.py
  ```
  ONNX theke YuNet/WHENet convert korte:
  ```bash
  python tools/export_onnx_rknn.py --model models/yunet.onnx \
    --out models/yunet-rk3576.rknn
  python tools/export_onnx_rknn.py --model models/whenet.onnx \
    --out models/whenet-rk3576.rknn
  ```
  Note: YOLO11 ONNX/RKNN output-e class probability agei sigmoid kora thake;
  app abar sigmoid kore na. onnx input naam `images`,
  `[1,3,H,W]` — script eguloi dhore rakhe.
- YuNet face NMS/box decode ebong model-er image pre/postprocessing CPU-te hoy;
  neural network inference NPU-te hoy.
- WHENet fallback-e `HEADPOSE_PROVIDER=auto` diye ONNX provider select hoy.
- MiVOLO bg thread e chole — infer loop kokhono 1s block hoy na;
  natun face e 1 sec `Unknown` theke result ase.
- `/api/health` e `hw` (dri/rknpu list), `face_backend`,
  `mivolo_backend`, `headpose_backend`, `npu_active` dekha jay.

## Speed tuning (board slow hole)
- `TORCH_THREADS=2 ORT_THREADS=1` (default) — 8-core board e thread fight komay.
  PC te (12+ core) `4/2` barate paro.
- `MIVOLO_REFRESH=3.0` — bg worker load komay (age update 3s por por).
- `IMG_SIZE=256` — YOLO ~40% fast, accuracy ektu kombe.
- `DETECT_EVERY=3 FACE_EVERY=3` — box update kom, stream eki thakbe.
- `MIVOLO_ENABLE=0` — sobcheye vari model off (gender/age lagbe na hole).
- 1 browser tab e dekho — proti tab alada JPEG encode kore (N tab = Nx load).

## RK3576 note
- Base image `python:3.11-slim` → arm64/amd64 duitai chole, tai same Dockerfile board + PC te cholbe.
- Stream ~25fps constant; box update CPU onujayi ~2-8fps (thread decoupled).
- RKNN NPU model path: YOLO exporter `models/yolo11n-rk3576.rknn` toiri kore;
  MiVOLO age/gender exporter `models/mivolo_v2_rk3576.rknn` toiri kore.
  YuNet face detection ebong WHENet head pose ekhono ONNX backend-e chole.
- Camera permission issue hole `privileged: true` uncomment koro ba `sudo usermod -aG video $USER`.

## Dev (docker chara)
```bash
pip install -r requirements.txt
CAMERA_SOURCE=0 IMG_SIZE=480 uvicorn app.main:app --host 0.0.0.0 --port 8080
```
