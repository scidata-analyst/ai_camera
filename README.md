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
# docker-compose.yml e devices line comment kore:
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
| MODEL_NAME | yolo11n.pt | `yolo11n/s/m` |
| CONF_THRESHOLD | 0.5 | 0.3-0.7 |
| IMG_SIZE | 384 | 320=fast, 480=accurate, 384=balance (32 er multiple) |
| FRAME_WIDTH/HEIGHT | 640x480 | camera resolution |
| JPEG_QUALITY | 70 | stream quality |
| SMOOTH_WINDOW | 1.0 | 1 sec window te mean/median/mode, stable = mode |
| FACE_ENABLE | 1 | 0 korle face off |
| FACE_SCORE | 0.6 | YuNet conf threshold |
| FACE_MODEL | /app/models/yunet.onnx | auto-download hoy |
| FACE_MIN_SIZE | 60 | choto face e gender Unknown (vul prediction komate) |
| MIVOLO_ENABLE | 1 | 0 korle age/gender off |
| MIVOLO_MODEL | iitolstykh/mivolo_v2 | HF cache e download (~115MB) |
| MIVOLO_REFRESH | 1.0 | proti sec e re-check -> sec proti median (barale CPU kombbe) |
| MIVOLO_BLUR | 80 | er niche sharpness hole age refresh off (motion blur fix) |
| HEADPOSE_ENABLE | 1 | 0 korle looking off |
| HEADPOSE_MODEL | /app/models/whenet.onnx | auto-download (PINTO0309, 17MB) |
| LOOK_YAW / LOOK_PITCH | 25 / 20 | looking threshold degree |

## Speed design (lag-free)
- Inference background thread e chole, MJPEG stream alada ~25-30fps e serve hoy.
- Tai YOLO slow holeo video atke na — box update infer rate e (~2-8fps) hoy.
- `IMG_SIZE` komale box update fast, barale accuracy valo.

## Looking at camera (WHENet head pose)
- Per face: `yaw/pitch/roll` degree + `looking` bool.
- `|yaw|<=25° and |pitch|<=20°` → **Looking** (green box + 👀 label).
- `/api/faces` e full JSON; header e `Look:x` count. CPU te ~15ms/face.

## Gender + Age (MiVOLO V2)
- Per face: `gender Male/Female + gconf` + `age` (bochor). Adience gender 97.3.
- Input dual: YuNet face crop + YOLO human box (body); body na pele face-only.
- Video label: `M0.91 27y` / `F0.93 24y`; `/api/faces` e full JSON; header e `M:x W:y` count.
- Note: MiVOLO vari (28.7M, ~1s/face CPU) — tai face cache ache: ekoi face e
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

## RK3576 note
- Base image `python:3.11-slim` → arm64/amd64 duitai chole, tai same Dockerfile board + PC te cholbe.
- Stream ~25fps constant; box update CPU onujayi ~2-8fps (thread decoupled).
- NPU (RKNN) lagle `yolo11n.pt → .rknn` + YuNet/Gender INT8 convert korte hobe.
- Camera permission issue hole `privileged: true` uncomment koro ba `sudo usermod -aG video $USER`.

## Dev (docker chara)
```bash
pip install -r requirements.txt
CAMERA_SOURCE=0 IMG_SIZE=480 uvicorn app.main:app --host 0.0.0.0 --port 8080
```
