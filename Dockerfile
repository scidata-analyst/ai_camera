FROM python:3.11-slim
ARG TARGETARCH
ARG RKNN_VERSION=2.5.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_NAME=/app/models/yolo11n.pt \
    YOLO_BACKEND=rknn \
    RKNN_MODEL=/app/models/yolo11n-rk3576.rknn \
    RKNPU_CORE=AUTO \
    CONF_THRESHOLD=0.5 \
    PERSON_CONF=0.65 \
    IMG_SIZE=320 \
    JPEG_QUALITY=95 \
    STREAM_FPS=25 \
    DETECT_EVERY=2 \
    FACE_EVERY=1 \
    FACE_BACKEND=rknn \
    FACE_MODEL=/app/models/RetinaFace_mobile320.onnx \
    FACE_RKNN_MODEL=/app/models/RetinaFace_mobile320-rk3576.rknn \
    AGE_GENDER_BACKEND=rknn \
    AGE_GENDER_MODEL=/app/models/mivolo-face.onnx \
    AGE_GENDER_RKNN_MODEL=/app/models/mivolo-face-rk3576.rknn \
    AGE_GENDER_ALIGN=arcface \
    AGE_RKNN_MODEL=/app/models/dlib-age-rk3576.rknn \
    HEADPOSE_BACKEND=rknn \
    HEADPOSE_RKNN_MODEL=/app/models/whenet-rk3576.rknn \
    OPENCV_OPENCL=auto \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    FACE_MIN_SIZE=60 \
    CAMERA_SOURCE=0 \
    FACE_ENABLE=1 \
    FACE_SCORE=0.6 \
    HEADPOSE_ENABLE=1 \
    HEADPOSE_MODEL=/app/models/whenet.onnx \
    LOOK_YAW=25 \
    LOOK_PITCH=20

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgomp1 ca-certificates \
    ocl-icd-libopencl1 mesa-opencl-icd \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip "setuptools<81" wheel && \
    pip install --retries 5 --timeout 120 -r requirements.txt && \
    if [ "$(uname -m)" = "aarch64" ] || [ "$TARGETARCH" = "arm64" ]; then \
      pip install "rknn-toolkit-lite2==${RKNN_VERSION}" && \
      python -c "from urllib.request import urlretrieve; urlretrieve('https://raw.githubusercontent.com/airockchip/rknn-toolkit2/v${RKNN_VERSION}/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so', '/usr/lib/librknnrt.so')"; \
    else \
      echo "[docker] x86 build: rknn-toolkit-lite2 skipped"; \
    fi && \
    rm -rf /root/.cache/pip /tmp/* /var/tmp/*

COPY app/ ./app/

RUN ls -lh /app/ || true

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
