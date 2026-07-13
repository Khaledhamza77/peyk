# peyk-vllm-paddleocr

Persistent vLLM server for PaddleOCR-VL-0.9B. `peyk-paddleocr-vl` (the OCR stage container)
is an HTTP client against this; it does not run inside `peyk-orchestrator`'s normal per-stage
`docker run --rm` lifecycle (see `stages.py`) because it needs to stay warm across runs
instead of paying model-load cost every invocation.

## Start

```
./start.sh
```

Starts (or restarts) the server on the `peyk-net` Docker network, reachable from other
containers on that network at `http://peyk-vllm-paddleocr:8118/v1`. Model weights are cached
in their own `peyk-vllm-paddleocr-cache` volume — this image runs as a non-root `paddleocr`
user (home `/home/paddleocr`, not `/root`), so it can't share the `peyk-paddlex-cache` volume
the other, root-running Paddle containers use.

## Stop

```
docker kill peyk-vllm-paddleocr
```

## Verify it's up

```
docker logs -f peyk-vllm-paddleocr
curl http://localhost:8118/v1/models
```

## Why this exists

PaddleX's local (in-process) `create_model()` path for PaddleOCR-VL has three structural
problems that aren't fixable via config — see `docs/build_notes.md`'s "`paddleocr-vl`
reliability investigation": a naive token-by-token generate loop that syncs GPU->CPU every
step (near-idle GPU, one CPU thread pegged), batch size hardcoded to 1, and no way to pass
decode guards like `repetition_penalty` to suppress its occasional runaway-repetition failure
mode. This server replaces that local path with vLLM's own decode loop (continuous batching,
CUDA graphs) and exposes those decode-guard params over an OpenAI-compatible API that
PaddleX's built-in `DocVLMGenAIClientPredictor` already knows how to speak to (via
`engine_config={"backend": "vllm-server", ...}` — no bespoke HTTP client needed).

## GPU memory

`vllm_config.yml` caps `gpu_memory_utilization` at **0.7**, leaving the rest of the card free
for `peyk-layout`/`peyk-simple-ocr`/`peyk-tsr`'s per-stage `--gpus all` runs. 0.35 was tried
first and failed outright: vLLM's own `torch.compile`/CUDA-graph capture for this model already
exceeds a 0.35 budget before any KV cache is allocated ("Available KV cache memory: -1.98 GiB").
Adjust down only after also shrinking `max_model_len`/`max_num_batched_tokens` (see
`vllm_config.yml`'s own comments) if 0.7 alone isn't enough headroom to share.
