# No CUDA/GPU needed in this image: it's an HTTP client against the persistent
# peyk-vllm-paddleocr server, which does the actual GPU inference (see backends/paddleocr_vl.py).
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Skips PaddleX's per-load connectivity probe to its model hoster — harmless here since this
# container never downloads model weights itself, but paddlex checks it unconditionally at
# import time regardless of engine backend.
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

COPY backends/ backends/
COPY run.py .

ENTRYPOINT ["python3", "run.py"]
