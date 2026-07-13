# No GPU calls of its own in this image: it's an HTTP client against the persistent
# peyk-vllm-surya server (see that container's README), which does the actual GPU inference —
# same split as peyk-paddleocr-vl/peyk-vllm-paddleocr. That said, surya-ocr (via transformers)
# has a hard, unconditional dependency on torch — see requirements.txt's own comment; there's
# no way to install surya-ocr without pip also installing torch, regardless of which predictor
# classes this container's code actually calls.
FROM python:3.10-slim

WORKDIR /app

# Installed explicitly, before requirements.txt, from PyTorch's CPU-only wheel index. Left to
# pip's default behavior, `pip install torch` on Linux resolves to the CUDA-bundled build —
# nvidia-cublas-cu12, nvidia-cudnn-cu12, nvidia-cuda-runtime-cu12, and several more, multiple
# GB combined — even though this container never calls `.cuda()` or touches a GPU. Pinning
# this index instead installs the CPU-only wheel with none of those nvidia-*-cu12 packages.
# Unversioned deliberately: letting pip pick whatever CPU build satisfies transformers'
# version constraint (see the build log surya-ocr's resolution went through) is simpler than
# hand-verifying a specific torch version against transformers' own compatibility range.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# surya-ocr's RecognitionPredictor, when called with no layout boxes (our --stage ocr's usage —
# a single already-cropped text-region/table-cell image, not a full page), treats that as "no
# layout hint" and requests settings.SURYA_MAX_TOKENS_FULL_PAGE generated tokens per call.
# Originally overridden to 4096 here because the *old* peyk-vllm-surya config
# (--max-model-len=8000) couldn't fit Surya's own default (12288) at all — failed every single
# --stage ocr request with a 400 ("max_tokens=12288 cannot be greater than max_model_len=8000"),
# and surya-ocr's own retry wrapper swallowed that error after 4 attempts and returned a
# fallback {"text": "", "score": 1.0} instead of raising, so an overnight peyk-orchestrator run
# silently produced hundreds of empty cells before this was caught. Now that peyk-vllm-surya's
# --max-model-len is 18000 (raised alongside a --mm-processor-kwargs max_pixels cap, which
# separately bounds image-token cost regardless of source render resolution — see start.sh),
# Surya's own default (12288) fits comfortably again, including for fullpage mode's
# predict_recognition() call on a genuine whole page (see run_fullpage() in run.py) — the case
# 4096 was flagged as likely too small for. No longer overridden here, so a future surya-ocr
# upgrade's own default just applies automatically instead of silently staying pinned to 4096.

# Governs settings.SURYA_MAX_TOKENS_TABLE_REC, used by TableRecPredictor's structure-only
# `predict()` method (--stage tsr's predict_table_structure) — NOT predict_full, see below.
# Bumped defensively above Surya's own default (3072) when the other two overrides were added;
# not driven by the old --max-model-len=8000 ceiling the way those two were (3072 and 4096 both
# fit comfortably under 8000 already), so raising --max-model-len to 18000 doesn't change the
# reasoning here — left as-is rather than reverted, since no truncation was ever observed on
# this path either way.
ENV SURYA_MAX_TOKENS_TABLE_REC=4096

# --stage table-full (predict_full) truncated every real table sent through it — turned out
# to be a completely different bug than SURYA_MAX_TOKENS_TABLE_REC above, confirmed by reading
# table_rec/__init__.py's actual source: predict_full sizes max_tokens via
# image_token_budget(count, ceiling=SURYA_MAX_TOKENS_BLOCK_CEILING, floor=1024) — a *different*
# setting than TABLE_REC, which explains why raising that one first had zero effect. `count`
# comes from predict_full's own `counts` parameter, which backends/client.py's
# predict_table_full was never passing — defaulting to 0 per table, clamping every request
# down to the floor (1024 tokens) regardless of the table's real size. Fixed on the client
# side (client.py now always passes a deliberately large count, reliably hitting the ceiling
# instead of the floor — see that file's own comment for why a real per-table count isn't used
# instead). This env var was the other half of that fix: Surya's own default ceiling (8192)
# exceeded the *old* peyk-vllm-surya --max-model-len=8000 outright, so it was lowered here to
# leave real room for input image + prompt tokens on top of it. Now that --max-model-len is
# 18000, 8192 fits comfortably again — no longer overridden here, same reasoning as
# SURYA_MAX_TOKENS_FULL_PAGE above (a larger real table than any tested so far could still hit
# the old 6000 ceiling and truncate; 8192 gives more headroom before that risk resurfaces).

COPY backends/ backends/
COPY run.py .

ENTRYPOINT ["python3", "run.py"]
