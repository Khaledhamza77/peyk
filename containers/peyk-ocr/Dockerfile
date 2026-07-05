# Same base as peyk-layout (Task 1.1/1.2) so behavior is consistent across local dev and
# the Compute Engine target, and so the already-solved paddlepaddle-gpu/CUDA pin carries over.
FROM nvidia/cuda:12.9.2-base-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# paddlepaddle-gpu isn't on the default PyPI index in a CUDA-matched build; install it
# from Paddle's own index first, then the rest of the requirements.
RUN pip install --no-cache-dir paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# libgomp1 (GNU OpenMP runtime) is required by paddle's compiled core but wasn't in the
# base apt-get above; installed as its own layer after paddlepaddle-gpu so the earlier,
# multi-GB paddlepaddle-gpu layer stays cached instead of rebuilding on every change here.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# torch/torchvision for the Granite-Docling-258M backend (transformers-based, no PaddleX
# involvement). Pinned explicitly from PyTorch's cu126 index, same versions and same
# reasoning as peyk-layout Task 1.2: leaving either unpinned lets pip resolve a torchvision
# build without the +cu126 local version tag, which is ABI-incompatible with the cu126 torch
# build ("operator torchvision::nms does not exist").
RUN pip install --no-cache-dir torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu126

# Tesseract binary + language data for the Tesseract backend (CPU-only — mainline Tesseract
# has no GPU acceleration path). tesseract-ocr-eng is pulled in automatically as a dependency
# of tesseract-ocr; tesseract-ocr-ara is added explicitly for Arabic.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-ara \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backends/ backends/
COPY run.py .

ENTRYPOINT ["python3", "run.py"]
