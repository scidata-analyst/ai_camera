# AI Camera — Human + Face Detector (RK3576 + Docker + YOLOv11n + YuNet)

USB webcam (`/dev/video0`) theke frame niye YOLOv11n + YuNet diye detect kore.
- `person` (class 0) -> **Human** (green box)
- onno sob class -> **Non-Human** (orange box)
- YuNet -> **Face** (cyan box + 5 landmark dot: eyes, nose, mouth)
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
| IMG_SIZE | 480 | RK3576 CPU te 480 fast, PC te 640 |
| FRAME_SKIP | 2 | 2 mane proti 2nd frame e infer |
| FRAME_WIDTH/HEIGHT | 640x480 | camera resolution |
| JPEG_QUALITY | 75 | stream quality |
| SMOOTH_WINDOW | 1.0 | 1 sec window te mean/median/mode, stable = mode |
| FACE_ENABLE | 1 | 0 korle face off |
| FACE_SCORE | 0.6 | YuNet conf threshold |
| FACE_MODEL | /app/models/yunet.onnx | auto-download hoy |

## Face (YuNet) result
- Per face: `box [x1,y1,x2,y2]` + `conf` + `landmarks[5]` (eyes, nose, mouth corners).
- `/api/faces` e JSON, video te cyan box + red dot.
- Smoothing face er jonneo 1-sec mean/median/mode hoy.

## Smoothing (1 sec update)
- Proti frame er `human/non_human` count 1-sec sliding window e joma hoy.
- Proti 1 sec por por `mean / median / mode` publish hoy, UI + `/api/stats` e dekha jay.
- Video overlay: `Human:instant[stable:X]` + 2nd line `1s avg mean/median/mode`.
- Web UI panel o 1 sec por por auto-refresh hoy.

## RK3576 note
- Base image `python:3.11-slim` → arm64/amd64 duitai chole, tai same Dockerfile board + PC te cholbe.
- CPU mode e ~8-15 FPS (480p, FRAME_SKIP=2).
- NPU (RKNN) lagle pore `yolo11n.pt → .rknn` convert kore detector swap korte hobe.
- Camera permission issue hole `privileged: true` uncomment koro ba `sudo usermod -aG video $USER`.

## Dev (docker chara)
```bash
pip install -r requirements.txt
CAMERA_SOURCE=0 IMG_SIZE=480 uvicorn app.main:app --host 0.0.0.0 --port 8080
```
