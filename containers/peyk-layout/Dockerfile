# Same base as the local GPU environment validated in Task 1.1 and the Compute Engine
# target in Phase 2/3, so behavior is consistent across local dev and cloud deploy.
# Framework GPU wheels (paddlepaddle-gpu, torch, ...) bundle their own CUDA/cuDNN
# runtime libs via pip deps, so the "base" (non-devel, non-cudnn) tag is sufficient —
# only the NVIDIA driver + container toolkit need to be present on the host.
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

# torch/torchvision are transitive deps of doclayout-yolo and docling-ibm-models; installed
# together explicitly (own layer, cached independently of requirements.txt) from PyTorch's
# cu126 index. Both versions must be pinned explicitly - leaving either unpinned (even
# with --index-url set) lets pip resolve a torchvision build without the +cu126 local
# version tag, which is ABI-incompatible with the cu126 torch build
# ("operator torchvision::nms does not exist").
RUN pip install --no-cache-dir torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu126

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backends/ backends/
COPY run.py .

ENTRYPOINT ["python3", "run.py"]
