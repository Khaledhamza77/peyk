# Merged worker (docs-personal/new_containerization_strategy.md) — layout+tsr+ocr's three
# source Dockerfiles (peyk-layout, peyk-tsr, peyk-simple-ocr) were already near-identical: same
# CUDA base, same paddlepaddle-gpu==3.3.0 + torch==2.12.1/torchvision==0.27.1 cu126 pin, same
# apt deps, differing only in which stage's backends/ + weight-baking steps got copied in.
# dcr/vlm/surya/paddleocr-vl/orchestrator (former peyk-dcr, peyk-vlm, peyk-surya,
# peyk-paddleocr-vl, peyk-orchestrator) need no local GPU work of their own (pure pypdfium2 text
# extraction, or thin HTTP clients against remote cloud APIs / the still-separate
# peyk-vllm-server sidecar, or pure dispatch logic) but share this same image rather than a
# separate GPU/non-GPU split — see the design doc for why one merged worker, not several, is the
# target shape. The surya stage reuses the GPU torch already installed below for the other
# stages (its own original image installed a CPU-only build instead, since it had no other torch
# to reuse) — one fewer multi-hundred-MB duplicate install. --stage at runtime (see run.py,
# default "orchestrator") picks which stage actually executes; stages dispatch each other
# in-process now (stage_dispatch.py) rather than via docker-outside-of-docker, so this image no
# longer needs the docker CLI or host docker socket at all.
FROM nvidia/cuda:12.9.2-base-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-ara \
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

# torch/torchvision are transitive deps of doclayout-yolo, docling-ibm-models (heron/
# tableformer), transformers (tatr), and the easyocr backend. Installed together explicitly
# (own layer, cached independently of requirements.txt) from PyTorch's cu126 index. Both
# versions must be pinned explicitly - leaving either unpinned (even with --index-url set)
# lets pip resolve a torchvision build without the +cu126 local version tag, which is
# ABI-incompatible with the cu126 torch build ("operator torchvision::nms does not exist").
RUN pip install --no-cache-dir torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu126

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Skips PaddleX's per-load connectivity probe to its model hoster — a repeated latency cost on
# every container start. PP-DocLayoutV2/PP-StructureV3's weights themselves aren't baked into
# the image at build time (tried; paddlepaddle-gpu hard-requires libcuda.so.1 just to import,
# even for a "cpu" device, and `docker build` has no GPU passthrough) — instead mount a
# persistent volume at /root/.paddlex when running (see run_local.sh) so weights survive across
# runs after the first download.
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# Governs settings.SURYA_MAX_TOKENS_TABLE_REC, used by TableRecPredictor's structure-only
# predict() method (the surya stage's --role tsr) — NOT predict_full (--role table-full), which
# is sized differently (see stages/surya/backends/client.py's predict_table_full docstring).
# Bumped defensively above Surya's own default (3072); peyk-vllm-surya's --max-model-len is
# large enough that this was never observed to truncate either way, kept as-is from the original
# peyk-surya/Dockerfile rather than re-derived.
ENV SURYA_MAX_TOKENS_TABLE_REC=4096

# Heron and DocLayout-YOLO (layout) have no such constraint: they're plain huggingface_hub
# downloads (HTTP only, no GPU/paddle import involved), so their weights CAN be baked in here —
# meaning no runtime cache volume is needed (or mounted) for these two; a volume mounted at
# ~/.cache/huggingface at runtime would shadow this baked-in layer entirely and silently defeat
# the point of doing this at build time. Matching the exact repo_id/filename
# stages/layout/backends/heron.py and doclayout_yolo.py request at runtime, so those calls hit
# this pre-populated cache and make no network call at all.
RUN python3 -c "\
from huggingface_hub import hf_hub_download, snapshot_download; \
hf_hub_download(repo_id='juliozhao/DocLayout-YOLO-DocStructBench', filename='doclayout_yolo_docstructbench_imgsz1024.pt'); \
snapshot_download(repo_id='docling-project/docling-layout-heron')"

# TATR and TableFormer (tsr) have no such constraint either: TATR's HF weights load via a plain
# transformers `from_pretrained` call, and TableFormer's weights are a plain huggingface_hub
# `snapshot_download` — both baked in here, same reasoning as Heron/DocLayout-YOLO above.
# RapidTable's weights also download via plain HTTP (onnxruntime-based, no paddle/GPU
# dependency) and are baked in the same way. allow_patterns scopes the TableFormer download to
# only the tableformer/accurate subtree of ds4sd/docling-models (which otherwise bundles
# artifacts for every Docling model family) — must match
# stages/tsr/backends/tableformer.py's own snapshot_download call exactly, or the runtime call
# falls outside this baked-in cache and silently re-downloads.
RUN python3 -c "\
from transformers import AutoImageProcessor, TableTransformerForObjectDetection; \
AutoImageProcessor.from_pretrained('microsoft/table-transformer-structure-recognition'); \
TableTransformerForObjectDetection.from_pretrained('microsoft/table-transformer-structure-recognition')"

RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='ds4sd/docling-models', allow_patterns=['model_artifacts/tableformer/accurate/*'])"

RUN python3 -c "\
from rapid_table import RapidTable, RapidTableInput; \
from rapid_table.utils import ModelType; \
RapidTable(RapidTableInput(model_type=ModelType.SLANETPLUS, use_ocr=False))"

# EasyOCR (ocr) downloads its own detector (CRAFT) + recognizer weights the first time a
# Reader() is constructed — plain HTTP, no GPU import involved, so this CAN be baked in at
# build time too. Baked for both lang lists easyocr_backend.py supports (arabic: ["ar","en"],
# latin: ["en"]) so neither --lang option re-downloads at runtime. gpu=False here only affects
# this throwaway build-time object, not the runtime Reader (which still respects `device`
# normally); no runtime cache volume is mounted over ~/.EasyOCR, so this baked-in layer isn't
# at risk of being shadowed.
RUN python3 -c "\
import easyocr; \
easyocr.Reader(['ar', 'en'], gpu=False); \
easyocr.Reader(['en'], gpu=False)"

# Separate COPY per stage (not one `COPY stages/ stages/`) so editing one stage's code doesn't
# invalidate the other stages' cached layer.
COPY stages/layout stages/layout
COPY stages/tsr stages/tsr
COPY stages/ocr stages/ocr
COPY stages/dcr stages/dcr
COPY stages/vlm stages/vlm
COPY stages/surya stages/surya
COPY stages/paddleocr-vl stages/paddleocr-vl
COPY stages/orchestrator stages/orchestrator
COPY stage_dispatch.py .
COPY run.py .

ENTRYPOINT ["python3", "run.py"]
