# Same base/pattern as peyk-layout's Dockerfile (see that file's comments for the full
# rationale) — same CUDA base, same paddlepaddle-gpu / torch+torchvision split, since this
# container needs both ecosystems too: PP-StructureV3's table module (paddle) plus
# TATR/TableFormer (torch-based, via transformers / docling-ibm-models).
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

# torch/torchvision are transitive deps of transformers (TATR) and docling-ibm-models
# (TableFormer); installed together explicitly (own layer, cached independently of
# requirements.txt) from PyTorch's cu126 index. Both versions must be pinned explicitly -
# leaving either unpinned (even with --index-url set) lets pip resolve a torchvision build
# without the +cu126 local version tag, which is ABI-incompatible with the cu126 torch build
# ("operator torchvision::nms does not exist") — same gotcha hit in peyk-layout.
RUN pip install --no-cache-dir torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu126

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Skips PaddleX's per-load connectivity probe to its model hoster. PP-StructureV3's
# SLANet_plus weights aren't baked into the image at build time, same reason as
# PP-DocLayoutV2 in peyk-layout: paddlepaddle-gpu hard-requires libcuda.so.1 just to import,
# even for a "cpu" device, and `docker build` has no GPU passthrough — instead relies on the
# persistent /root/.paddlex volume peyk-orchestrator's stages.py already mounts for every
# stage (PADDLEX_CACHE_VOLUME), so weights survive across `--rm` runs after the first
# download, same as PP-DocLayoutV2 does today.
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# TATR and TableFormer have no such constraint: TATR's HF weights load via a plain
# transformers `from_pretrained` call (HTTP only, no GPU import involved), and TableFormer's
# weights are a plain huggingface_hub `snapshot_download` — both CAN be baked in here, same
# reasoning as Heron/DocLayout-YOLO in peyk-layout's Dockerfile. No runtime cache volume is
# mounted for either (that would shadow this baked-in layer and defeat the point).
# RapidTable's weights also download via plain HTTP (onnxruntime-based, no paddle/GPU
# dependency) and are baked in the same way.
RUN python3 -c "\
from transformers import AutoImageProcessor, TableTransformerForObjectDetection; \
AutoImageProcessor.from_pretrained('microsoft/table-transformer-structure-recognition'); \
TableTransformerForObjectDetection.from_pretrained('microsoft/table-transformer-structure-recognition')"

# allow_patterns scopes this to only the tableformer/accurate subtree of ds4sd/docling-models
# (which otherwise bundles artifacts for every Docling model family) — must match
# backends/tableformer.py's own snapshot_download call exactly, or the runtime call falls
# outside this baked-in cache and silently re-downloads (same gotcha noted in peyk-layout's
# Dockerfile for Heron/DocLayout-YOLO).
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='ds4sd/docling-models', allow_patterns=['model_artifacts/tableformer/accurate/*'])"

RUN python3 -c "\
from rapid_table import RapidTable, RapidTableInput; \
from rapid_table.utils import ModelType; \
RapidTable(RapidTableInput(model_type=ModelType.SLANETPLUS, use_ocr=False))"

COPY backends/ backends/
COPY run.py .

ENTRYPOINT ["python3", "run.py"]
