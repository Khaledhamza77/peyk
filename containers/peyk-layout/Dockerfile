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

# Skips PaddleX's per-load connectivity probe to its model hoster — a repeated latency
# cost on every container start. PP-DocLayoutV2's weights themselves aren't baked into the
# image at build time (tried; paddlepaddle-gpu hard-requires libcuda.so.1 just to import,
# even for a "cpu" device, and `docker build` has no GPU passthrough) — instead mount a
# persistent volume at /root/.paddlex when running (see peyk-orchestrator/stages.py) so
# weights survive across `--rm` runs after the first download.
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# Unlike PP-DocLayoutV2 above, Heron and DocLayout-YOLO have no such constraint: they're
# plain huggingface_hub downloads (HTTP only, no GPU/paddle import involved), so their
# weights CAN be baked in here — meaning no runtime cache volume is needed (or mounted) for
# these two; a volume mounted at ~/.cache/huggingface at runtime would shadow this baked-in
# layer entirely and silently defeat the point of doing this at build time. Matching the
# exact repo_id/filename backends/heron.py and backends/doclayout_yolo.py request at runtime,
# so those calls hit this pre-populated cache and make no network call at all.
RUN python3 -c "\
from huggingface_hub import hf_hub_download, snapshot_download; \
hf_hub_download(repo_id='juliozhao/DocLayout-YOLO-DocStructBench', filename='doclayout_yolo_docstructbench_imgsz1024.pt'); \
snapshot_download(repo_id='docling-project/docling-layout-heron')"

COPY backends/ backends/
COPY run.py .

ENTRYPOINT ["python3", "run.py"]
