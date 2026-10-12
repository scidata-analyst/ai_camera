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
    HF_HOME=/app/models/huggingface \
    HUGGINGFACE_HUB_CACHE=/app/models/huggingface \
    TRANSFORMERS_CACHE=/app/models/huggingface

# headless opencv -> libgl1/libdrm2/curl lagbe na.
# git sudhu MiVOLO build er jonno, pore purge hobe.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgomp1 ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip "setuptools<81" wheel && \
    if [ "$(uname -m)" = "aarch64" ] || [ "$TARGETARCH" = "arm64" ]; then \
      pip install --retries 5 --timeout 120 --index-url https://download.pytorch.org/whl/cpu torch; \
    fi && \
    pip install --retries 5 --timeout 120 -r requirements.txt && \
    (pip uninstall -y opencv-python || true) && \
    pip install --retries 5 --timeout 120 --no-deps --force-reinstall "opencv-python-headless>=4.9,<5.0" && \
    pip install --no-deps --no-build-isolation git+https://github.com/WildChlamydia/MiVOLO.git && \
    apt-get purge -y git && apt-get autoremove -y && rm -rf /root/.cache/pip /tmp/* /var/tmp/*
RUN if [ "$(uname -m)" = "aarch64" ]; then \
      pip install rknn-toolkit-lite2; \
    else \
      echo "[docker] x86 build: rknn-toolkit-lite2 skipped"; \
    fi

COPY app/ ./app/

# choto model (yolo 6MB + yunet 300KB + whenet 17MB) bake ok.
# boro MiVOLO (~115MB) runtime volume e cache hobe, image e bake na.
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')" \
    || echo "[warn] YOLO pre-download skip"
RUN mkdir -p /app/models/huggingface && \
    python -c "import urllib.request; urllib.request.urlretrieve('https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx','/app/models/yunet.onnx')" \
    || echo "[warn] yunet skip"; \
    python -c "import urllib.request; urllib.request.urlretrieve('https://github.com/PINTO0309/HeadPoseEstimation-WHENet-yolov4-onnx-openvino/releases/download/v1.0.4/whenet_1x3x224x224_prepost.onnx','/app/models/whenet.onnx')" \
    || echo "[warn] whenet skip"; \
    ls -lh /app/models/ || true

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
