FROM python:3.11-slim
ARG TARGETARCH

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
    HF_HOME=/app/models/huggingface \
    HUGGINGFACE_HUB_CACHE=/app/models/huggingface \
    TRANSFORMERS_CACHE=/app/models/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip "setuptools<81" wheel && \
    pip install --no-deps --retries 5 --timeout 120 --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 torchvision==0.23.0 && \
    pip install --retries 5 --timeout 120 -r requirements.txt && \
    pip install --no-deps --retries 5 --timeout 120 ultralytics==8.3.253 && \
    pip install --no-deps --no-build-isolation https://github.com/WildChlamydia/MiVOLO/archive/37475e3f8818b5f22448003feec3e64b01bfb188.tar.gz && \
    if [ "$(uname -m)" = "aarch64" ] || [ "$TARGETARCH" = "arm64" ]; then \
      pip install rknn-toolkit-lite2; \
    else \
      echo "[docker] x86 build: rknn-toolkit-lite2 skipped"; \
    fi && \
    rm -rf /root/.cache/pip /tmp/* /var/tmp/*

COPY app/ ./app/

RUN mkdir -p /app/models/huggingface && ls -lh /app/ || true

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
