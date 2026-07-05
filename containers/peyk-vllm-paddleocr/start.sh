#!/usr/bin/env bash
# Starts the persistent PaddleOCR-VL vLLM server that peyk-paddleocr-vl talks to over HTTP.
#
# This is NOT a per-stage `docker run --rm` like peyk-orchestrator's other stages (see
# stages.py) — it's a long-lived service, started once and left running, the same way you'd
# run a database sidecar. peyk-orchestrator does not manage its lifecycle.
#
# Uses PaddlePaddle's official prebuilt image rather than a Dockerfile we maintain: it already
# bundles a version of vLLM with PaddleOCR-VL's custom architecture (Ernie4.5 decoder +
# SigLIP-style vision encoder) registered, which mainline vLLM does not support out of the box.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

NETWORK=peyk-net
IMAGE=ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu

# On git-bash/MSYS, bare absolute-path CLI arguments (no colon in the token) get silently
# rewritten to a Windows path by the shell before docker ever sees them (e.g. `/tmp/foo`
# becomes `C:/.../Git/tmp/foo`) — this bit us repeatedly (chown target, --backend_config).
# MSYS_NO_PATHCONV disables that rewriting outright; `pwd -W` (also MSYS-specific, hence the
# `pwd` fallback for WSL/Linux) gives us the real Windows path for the one host bind-mount
# below explicitly, instead of relying on MSYS's own (otherwise reasonable) auto-translation
# of `-v host:container` arguments, which NO_PATHCONV would otherwise also suppress.
export MSYS_NO_PATHCONV=1
WIN_PWD=$(pwd -W 2>/dev/null || pwd)

docker network create "$NETWORK" 2>/dev/null || true
docker rm -f peyk-vllm-paddleocr 2>/dev/null || true

# A fresh named volume is created root-owned; this image runs as a non-root `paddleocr` user,
# which then can't write into it ("PermissionError: /home/paddleocr/.paddlex/temp"). Fix
# ownership as root once before the real (non-root) server container starts. Harmless/fast
# on subsequent runs once ownership is already correct.
docker run --rm --user root -v peyk-vllm-paddleocr-cache:/home/paddleocr/.paddlex \
  --entrypoint chown "$IMAGE" -R paddleocr:paddleocr /home/paddleocr/.paddlex

docker run -d --gpus all --network "$NETWORK" --name peyk-vllm-paddleocr \
  -e PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
  -v "$WIN_PWD/vllm_config.yml:/tmp/vllm_config.yml" \
  -v peyk-vllm-paddleocr-cache:/home/paddleocr/.paddlex \
  "$IMAGE" \
  paddleocr genai_server --model_name PaddleOCR-VL-0.9B \
  --host 0.0.0.0 --port 8118 --backend vllm --backend_config /tmp/vllm_config.yml

echo "peyk-vllm-paddleocr starting — follow logs with: docker logs -f peyk-vllm-paddleocr"
echo "Reachable from other containers on the '$NETWORK' network at http://peyk-vllm-paddleocr:8118/v1"
