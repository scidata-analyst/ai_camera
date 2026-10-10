# RK3576 (arm64) + x86_64 dual-arch compatible
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_NAME=yolo11n.pt \
    CONF_THRESHOLD=0.5 \
    IMG_SIZE=480 \
    FRAME_SKIP=2 \
    CAMERA_SOURCE=0

# opencv + torch dependency
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app/ ./app/

# model pre-download (build cache, runtime eo download hobe jodi na thake)
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
