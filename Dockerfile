# RK3576 (arm64) + x86_64 dual-arch compatible
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_NAME=yolo11n.pt \
    CONF_THRESHOLD=0.5 \
    IMG_SIZE=384 \
    JPEG_QUALITY=70 \
    FACE_MIN_SIZE=60 \
    CAMERA_SOURCE=0 \
    FACE_ENABLE=1 \
    FACE_SCORE=0.6 \
    FACE_MODEL=/app/models/yunet.onnx \
    MIVOLO_ENABLE=1 \
    MIVOLO_MODEL=iitolstykh/mivolo_v2 \
    HEADPOSE_ENABLE=1 \
    HEADPOSE_MODEL=/app/models/whenet.onnx \
    LOOK_YAW=25 \
    LOOK_PITCH=20 \
    HF_HOME=/root/.cache/huggingface

# opencv + torch dependency (+git for MiVOLO lib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    v4l-utils curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip "setuptools<81" wheel && pip install -r requirements.txt && \
    pip install --no-deps --no-build-isolation git+https://github.com/WildChlamydia/MiVOLO.git

COPY app/ ./app/

# model pre-download (build cache, runtime eo download hobe jodi na thake)
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
RUN mkdir -p /app/models && \
    curl -L -o /app/models/yunet.onnx \
    https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx && \
    curl -L -o /app/models/whenet.onnx \
    https://github.com/PINTO0309/HeadPoseEstimation-WHENet-yolov4-onnx-openvino/releases/download/v1.0.4/whenet_1x3x224x224_prepost.onnx && \
    ls -lh /app/models/
# MiVOLO weights (~115MB) pre-download -> HF cache
RUN python -c "import app.mivolo; from transformers import AutoModelForImageClassification; AutoModelForImageClassification.from_pretrained('iitolstykh/mivolo_v2', trust_remote_code=True)"

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
