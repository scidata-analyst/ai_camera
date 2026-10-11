# RK3576 (arm64) + x86_64 dual-arch compatible
FROM python:3.11-slim

# NPU: YOLO_BACKEND=auto|cpu|rknn. auto = .rknn file thakle NPU, naile CPU
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_NAME=yolo11n.pt \
    YOLO_BACKEND=auto \
    RKNN_MODEL=/app/models/yolo11n-rk3576.rknn \
    RKNPU_CORE=AUTO \
    CONF_THRESHOLD=0.5 \
    IMG_SIZE=320 \
    JPEG_QUALITY=60 \
    STREAM_FPS=25 \
    DETECT_EVERY=2 \
    MAX_FACES=3 \
    TORCH_THREADS=4 \
    ORT_THREADS=2 \
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    FACE_MIN_SIZE=60 \
    CAMERA_SOURCE=0 \
    FACE_ENABLE=1 \
    FACE_SCORE=0.6 \
    FACE_MODEL=/app/models/yunet.onnx \
    MIVOLO_ENABLE=1 \
    MIVOLO_MODEL=iitolstykh/mivolo_v2 \
    MIVOLO_REFRESH=2.0 \
    MIVOLO_MAX_FACES=1 \
    HEADPOSE_ENABLE=1 \
    HEADPOSE_MODEL=/app/models/whenet.onnx \
    LOOK_YAW=25 \
    LOOK_PITCH=20 \
    HF_HOME=/root/.cache/huggingface

# opencv + torch dependency (+git for MiVOLO lib)
# libdrm2 = RK3576 /dev/dri access er jonno (NPU/DRI node khulte lage)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libdrm2 \
    v4l-utils curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip "setuptools<81" wheel && pip install -r requirements.txt && \
    pip install --no-deps --no-build-isolation git+https://github.com/WildChlamydia/MiVOLO.git
# NPU runtime: sudhu ARM64 (RK3576 board) e install hobe.
# x86 PC te build fail korbe na — uname check kore skip kore.
RUN if [ "$(uname -m)" = "aarch64" ]; then \
      pip install rknn-toolkit-lite2; \
    else \
      echo "[docker] x86 build: rknn-toolkit-lite2 skipped (board e auto install hobe)"; \
    fi

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
