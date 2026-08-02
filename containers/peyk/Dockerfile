# Merged layout+tsr+ocr worker (docs-personal/new_containerization_strategy.md) — the three
# source Dockerfiles (peyk-layout, peyk-tsr, peyk-simple-ocr) were already near-identical: same
# CUDA base, same paddlepaddle-gpu==3.3.0 + torch==2.12.1/torchvision==0.27.1 cu126 pin, same
# apt deps, differing only in which stage's backends/ + weight-baking steps got copied in. This
# Dockerfile is the union of all three; --stage at runtime (see run.py) picks which one actually
# executes.
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
# persistent volume at /root/.paddlex when running (see peyk-orchestrator/stages.py) so weights
# survive across `--rm` runs after the first download.
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

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
# invalidate the other two stages' cached layer.
COPY stages/layout stages/layout
COPY stages/tsr stages/tsr
COPY stages/ocr stages/ocr
COPY run.py .

ENTRYPOINT ["python3", "run.py"]
