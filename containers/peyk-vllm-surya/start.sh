#!/usr/bin/env bash
# Starts the persistent Surya-OCR-2 vLLM server that peyk-surya talks to over HTTP.
#
# This is NOT a per-stage `docker run --rm` like peyk-orchestrator's other stages (see
# stages.py) — it's a long-lived service, started once and left running, the same way
# peyk-vllm-paddleocr is (see that container's start.sh for the identical rationale). Uses
# the official `vllm/vllm-openai` image directly (unlike peyk-vllm-paddleocr, which needs
# PaddlePaddle's own prebuilt image for a custom architecture) — confirmed live: vLLM v0.24.0
# resolves Surya-OCR-2 as `Qwen3_5ForConditionalGeneration`, a natively-supported architecture
# (trust_remote_code=False, no patched vLLM build needed), matching docs/surya/details.md's
# assumption. `curl :8119/v1/models` returns the model correctly with this config.
#
# Cold start takes ~14 minutes with the config below (see the GPU-memory tuning notes for
# why) — this is not a quick sidecar to restart casually.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

NETWORK=peyk-net
# Pinned to v0.20.1 — the exact version surya-ocr's own settings.py specifies
# (VLLM_DOCKER_IMAGE) as its tested/reference config, rather than `:latest` (which resolved to
# v0.24.0, four minor versions ahead). Worth testing directly: v0.24.0's "CUDA graph memory
# profiling enabled by default since v0.21.0" is the exact accounting behavior behind every
# deficit failure chased below — v0.20.1 predates it entirely, so DataLab's own reference
# config never had to contend with it. If this resolves the memory math cleanly, the earlier
# failures were partly a version-mismatch artifact, not a hard ceiling on this card.
IMAGE=vllm/vllm-openai:v0.20.1
MODEL=datalab-to/surya-ocr-2
PORT=8119

# See peyk-vllm-paddleocr/start.sh for why MSYS_NO_PATHCONV/pwd -W are needed on git-bash/MSYS
# (bare absolute-path CLI args otherwise get silently rewritten to a Windows path).
export MSYS_NO_PATHCONV=1

docker network create "$NETWORK" 2>/dev/null || true
docker rm -f peyk-vllm-surya 2>/dev/null || true

# vllm/vllm-openai runs as root by default (unlike PaddlePaddle's genai-vllm-server image),
# so no chown-fixup step is needed before mounting the HF cache volume here.
#
# torch.compile's own compile cache is written to /root/.cache/vllm, not
# /root/.cache/huggingface — mounted as its own named volume so it survives across
# `docker rm -f` + restart instead of being rebuilt from scratch every single time (moot with
# --enforce-eager below, which skips torch.compile/CUDA-graph capture entirely — kept anyway
# in case --enforce-eager is ever dropped once this card's headroom situation improves).
#
# GPU memory tuning below is the result of three real, escalating failures on this exact
# 12GB card (see docs/implementation_plan.md Task 1.8 / build_notes.md for the full log
# trail), not arbitrary defaults:
#   1. gpu_memory_utilization=0.7 (matching peyk-vllm-paddleocr) failed with "Available KV
#      cache memory: -2.69 GiB" — Surya-OCR-2's own baseline footprint (weights + activations
#      + encoder cache + CUDA-graph overhead) alone needs ~11.08GiB, ~92% of the whole card,
#      before any real request is served.
#   2. Raising to 0.95 failed differently, even earlier: this GPU also drives the host's
#      display, so only ~10.85GiB of the nominal ~11.99GiB is ever actually free — vLLM's own
#      pre-flight check refused to start before touching weights at all.
#   3. --enforce-eager (skips CUDA-graph capture, ~0.4GiB of that baseline) + 0.90 (the
#      largest fraction clearing the free-memory pre-flight check) was the combination that
#      finally got a POSITIVE "Available KV cache memory: 0.12 GiB" — this card genuinely
#      cannot spare more than that for this model while it's also driving a display, and 0.90
#      is very close to the practical ceiling; going lower reintroduces failure #1's deficit.
#
# --max-model-len caps the last piece: vLLM still refused to start even with that positive
# 0.12GiB, because serving one request at Surya-OCR-2's own default max_model_len (262144)
# alone needs 3.02GiB of KV cache. vLLM's own error told us the actual ceiling given 0.12GiB:
# "the estimated maximum model length is 8160". 8000 (a little under that) is used instead of
# capping to the exact estimate, as a small safety margin against run-to-run memory variance.
# This is also right-sized for the workload, not just a number that happens to fit: every call
# this pipeline sends is one page/region/cell image with a modest text/HTML output, never
# anything close to 262144 tokens — see peyk-surya/run.py's own module docstring.
#
# Consequence of all this: this sidecar needs near-exclusive use of the GPU while running —
# unlike peyk-vllm-paddleocr (0.7, ~8.4GB), it will NOT coexist with peyk-layout/
# peyk-simple-ocr/peyk-tsr's own per-stage `--gpus all` runs, or with peyk-vllm-paddleocr
# itself. Stop every other GPU consumer before starting this.
#
# Confirmed working (curl :8119/v1/models responded correctly), but with two real costs from
# --enforce-eager, neither yet optimized:
#   - ~14 minute cold start ("init engine (profile, create kv cache, warmup model) took
#     841.09 s") — counterintuitively SLOWER than the earlier (failed) compiled attempt's
#     profiling step, because eager-mode profiling runs raw, unoptimized PyTorch instead of a
#     compiled/CUDA-graphed path.
#   - "Maximum concurrency for 8,000 tokens per request: 1.00x" — effectively no real request
#     concurrency at the full token budget; keep this in mind before wiring up anything like
#     peyk-paddleocr-vl's concurrent ThreadPoolExecutor dispatch pattern for this server.
#
# EXPERIMENT (v0.24.0, superseded by the version pin above): --enforce-eager dropped, then
# VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 tried on top — both still failed with a small
# but real KV-cache deficit (-0.29GiB, then 0.11GiB available vs. 0.11GiB needed with zero
# margin for actual graph capture).
#
# EXPERIMENT (v0.20.1 pin above): confirmed via the startup banner this really is v0.20.1, yet
# it still logs "CUDA graph memory profiling is enabled (default since v0.21.0)" and reproduces
# the same ~-0.3GiB deficit — genuinely inconsistent with the version pin, unresolved, not worth
# chasing further today.
#
# EXPERIMENT (--max-num-seqs 1): every deficit so far came from vLLM's warmup profiling
# sizing activation/workspace memory for a much larger hypothetical concurrent load than this
# pipeline ever sends (our own client dispatches one request at a time, and the server's own
# "Maximum concurrency ... 1.00x" already confirms real concurrency tops out at 1 anyway).
# Unlike the version-pin/profiler-estimate experiments, this changes what vLLM actually
# profiles against, not just its accounting — directly analogous to the fix already applied in
# peyk-vllm-paddleocr/vllm_config.yml for a different model's oversized-default problem. Real
# result: "Estimated CUDA graph memory" dropped from 0.40GiB to 0.03GiB, and "Available KV
# cache memory" went from -0.3GiB to +0.08GiB — genuinely closed most of the gap, not just
# relocated it. Still came up just short: 0.11GiB needed vs 0.08GiB available (~0.03GiB short)
# at max_model_len=8000.
#
# EXPERIMENT (gpu_memory_utilization 0.90 -> 0.904): small, deliberate nudge, not a repeat of
# the earlier 0.93/0.95 attempts — those failed a DIFFERENT, earlier check (the free-memory
# pre-flight test, whose real ceiling on this display-sharing card is ~0.905 = 10.85GiB
# actually free / 11.99GiB total). 0.904 stayed just under that ceiling while adding enough to
# the KV-cache budget to close the earlier ~0.03GiB gap. Result: CONFIRMED WORKING —
# "Available KV cache memory: 0.13 GiB", real CUDA graphs captured, "init engine ... took
# 219.37 s" (vs. 841s under --enforce-eager), "Maximum concurrency for 8,000 tokens per
# request: 1.11x".
#
# EXPERIMENT (--max-num-seqs 1 -> 2): peyk-surya/run.py's --stage ocr dispatches crops through
# a ThreadPoolExecutor (concurrency 8 by default), but max-num-seqs=1 means the SERVER only
# ever admits one sequence at a time regardless of how many the client sends at once — that
# capped what the client-side concurrency change could actually do. Result: CONFIRMED WORKING,
# small cost — "Available KV cache memory: 0.11 GiB" (down from 0.13GiB at max-num-seqs=1,
# ~0.02GiB), CUDA graphs still captured fine (PIECEWISE=3/largest=4, FULL=2/largest=2),
# "Application startup complete."
#
# EXPERIMENT (--max-num-seqs 2 -> 4): FAILED — "Available KV cache memory: 0.11 GiB" needed
# vs. 0.11 GiB available (rounds to the same displayed value, but short in reality), estimated
# max feasible length 7616 vs. our 8000 requirement. The slightly larger CUDA graph capture
# range 4 needs (PIECEWISE largest=8 vs. 2's largest=4) was enough to tip it over. Reverted to
# 2 — the confirmed practical ceiling for max-num-seqs at max-model-len=8000 on this card;
# raising further needs either lowering max-model-len (the layout/tsr full-page-image tradeoff
# already avoided once) or more free memory than this display-sharing card has.
#
# EXPERIMENT (adopting surya-ocr's own VllmBackend defaults instead of hand-tuning further):
# surya/inference/backends/vllm.py — the code path this project bypasses by setting
# SURYA_INFERENCE_URL, but which a colleague's setup relies on unmodified (auto-spawning its
# own vLLM container instead of pointing at a persistent sidecar like this one) — carries
# defaults from surya/settings.py that differ from every value hand-tuned above:
#   VLLM_MAX_MODEL_LEN=18000 (ours: 8000), VLLM_GPU_MEMORY_UTILIZATION=0.85 (ours: 0.904), and
#   a `--mm-processor-kwargs '{"min_pixels": 3136, "max_pixels": 6291456}'` flag this file never
#   set at all. That last one matters most: it caps vision-token cost per image regardless of
#   source render resolution — without it, a large 300 DPI fullpage render has no ceiling on
#   how many tokens it can cost, which is the leading theory for why fullpage mode
#   (predict_recognition() on a whole page, not a small crop) was seen stuck indefinitely in
#   vLLM's Waiting queue with 0% KV cache usage (see implementation_plan.md Task 1.8) — a
#   request whose real input+output token cost can't be known until the multimodal processor
#   actually runs doesn't get vLLM's normal synchronous max_model_len rejection, so it can
#   silently deadlock instead of erroring. VLLM_GPU_TYPE (default "4090", a 24GB baseline used
#   only to scale max-num-seqs/max-num-batched-tokens in _gpu_settings()) is deliberately NOT
#   replicated here — this card is a 12GB RTX 3500 Ada, not a 4090, and max-num-seqs is set
#   directly below instead of via that GPU-scaled ratio.
EXTRA_VLLM_ARGS=()
if [ "${ENFORCE_EAGER:-0}" = "1" ]; then
  EXTRA_VLLM_ARGS+=(--enforce-eager)
fi

docker run -d --gpus all --network "$NETWORK" --name peyk-vllm-surya \
  --ipc=host \
  -v peyk-vllm-surya-cache:/root/.cache/huggingface \
  -v peyk-vllm-surya-torch-cache:/root/.cache/vllm \
  -p "$PORT:8000" \
  "$IMAGE" \
  --model "$MODEL" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
  --max-model-len "${MAX_MODEL_LEN:-18000}" \
  --max-num-seqs "${MAX_NUM_SEQS:-8}" \
  --mm-processor-kwargs '{"min_pixels": 3136, "max_pixels": 6291456}' \
  "${EXTRA_VLLM_ARGS[@]}"

echo "peyk-vllm-surya starting — follow logs with: docker logs -f peyk-vllm-surya"
echo "Reachable from other containers on the '$NETWORK' network at http://peyk-vllm-surya:8000/v1"
echo "Reachable from the host at http://localhost:$PORT/v1"
