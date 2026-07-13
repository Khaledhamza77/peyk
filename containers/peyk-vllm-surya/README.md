# peyk-vllm-surya

Persistent vLLM server for `datalab-to/surya-ocr-2`. `peyk-surya` (the layout/TSR/OCR client
container, Task 1.8) is an HTTP client against this; it does not run inside
`peyk-orchestrator`'s normal per-stage `docker run --rm` lifecycle (see `stages.py`), same
reasoning as `peyk-vllm-paddleocr` — model load / CUDA graph capture is slow enough that
paying it once and staying warm beats paying it on every stage invocation.

## Before relying on this

`docs/surya/details.md`'s core assumption — that stock `vllm/vllm-openai` serves this model
out of the box — is confirmed live, not just architecturally plausible: vLLM v0.24.0 resolves
it as `Qwen3_5ForConditionalGeneration`, a natively-supported architecture
(`trust_remote_code=False`, no patched vLLM build needed, unlike `peyk-vllm-paddleocr`), and
`curl :8119/v1/models` returns the model correctly with the config below. What's still
unconfirmed is the shape of its layout/recognition/table-rec API responses — see
`containers/peyk-surya/backends/client.py`'s module docstring.

## Smoke test

```
./start.sh
docker logs -f peyk-vllm-surya   # watch for "Application startup complete." — ~80s cold
                                  # (see GPU memory section below; this used to take ~14 minutes)
curl http://localhost:8119/v1/models
```

## Start

```
./start.sh
```

Starts (or restarts) the server on the `peyk-net` Docker network, reachable from other
containers at `http://peyk-vllm-surya:8000/v1` (mapped to `http://localhost:8119/v1` on the
host). Model weights are cached in their own `peyk-vllm-surya-cache` volume, and vLLM's
`torch.compile` cache (the biggest single startup-time cost after the very first run — see
below) in its own `peyk-vllm-surya-torch-cache` volume.

## Stop

```
docker kill peyk-vllm-surya
```

## GPU memory — resolved; current config is healthy, not tight

**Current working config**: `--gpu-memory-utilization 0.85`, `--max-model-len 18000`,
`--max-num-seqs 8`, `--mm-processor-kwargs '{"min_pixels": 3136, "max_pixels": 6291456}'` (see
`start.sh`). Confirmed live: `Available KV cache memory: 8.2 GiB`, `Maximum concurrency for
18,000 tokens per request: 35.57x`, engine init `79.42s` — real headroom, not a tuned-to-the-edge
squeeze.

This was not always true. Getting a *working* boot at all originally took three escalating real
failures on this 12GB, display-sharing card (only ~10.85GiB of the nominal ~11.99GiB is ever
actually free), landing on `gpu_memory_utilization=0.904` + `--enforce-eager` +
`--max-model-len=8000` — a config that only barely worked (`Available KV cache memory: 0.13
GiB`), needed near-exclusive GPU use, and had a ~14 minute cold start under `--enforce-eager`
(`~4min` once real CUDA graphs were later coaxed to fit instead). Full escalating trail kept in
`start.sh`'s own comments for the historical record.

**Root cause, found later**: none of that tuning was actually fighting a hard ceiling on this
card — it was working around one specific, fixable gap. Comparing against a colleague's
independent Surya setup (a thin CLI client + Dockerfile, running on hardware described as
identical to this one) showed the difference: his setup never sets `SURYA_INFERENCE_URL`, so
`surya-ocr`'s own `VllmBackend.start()` auto-spawns its vLLM server using the *library's own*
defaults (`surya/settings.py`) — `VLLM_MAX_MODEL_LEN=18000`, `VLLM_GPU_MEMORY_UTILIZATION=0.85`,
plus a `--mm-processor-kwargs max_pixels=6291456` flag this file never set at all. That last one
was the actual fix: without it, vLLM reserves an encoder-cache budget sized against an
*unbounded* hypothetical image — confirmed via the boot log (`Encoder cache will be initialized
with a budget of 6144 tokens` once capped, vs. no such bound before). That reservation, not
`gpu_memory_utilization` or `max_model_len` themselves, was silently eating nearly the entire
"available" memory pool before KV-cache sizing ever got a turn. Every earlier fix
(`--enforce-eager`, reducing `--max-num-seqs`, squeezing `gpu_memory_utilization` up to `0.904`)
was closing symptoms of this one uncapped reservation, not the reservation itself. `--max-num-seqs`
is kept at `8` rather than the library's own GPU-scaled default of `32` — that scaling assumes a
`VLLM_GPU_TYPE="4090"` 24GB baseline, and this is a 12GB card, so `8` is set directly instead
(also matching `peyk-surya`'s own client-side concurrency default — see that container's
`run.py`).

**Coexistence with `peyk-vllm-paddleocr`**: the old "will NOT coexist, needs near-exclusive GPU
use" guidance below is **confirmed resolved**, not just theorized — a real `run_local.sh` run
with `tsr: surya` (routing tables through `--stage table-full` against this server) and
`ocr: paddleocr-vl` (against `peyk-vllm-paddleocr`) dispatched both sidecars successfully in the
same pipeline run, no `Connection error`s. The identical config failed every `paddleocr-vl` crop
before the GPU-memory fix below. The commands below are now only a pre-fix-era fallback, not
current guidance:

```
docker kill peyk-vllm-paddleocr   # only if coexistence somehow regresses again
```

## Confirmed working

`Application startup complete.`, `curl :8119/v1/models` responds correctly, `~79s` cold start,
`8.2 GiB` KV cache, `35.57x` max concurrency at the full `18,000`-token budget. Unlike the old
config, concurrent client dispatch is now genuinely worthwhile — `peyk-surya/run.py`'s
`ThreadPoolExecutor` + `tqdm` pattern (originally `--stage ocr` only) has since been extended to
`--stage layout`/`tsr`/`table-full` and `--mode fullpage` too, all defaulting to concurrency `8`
to match `--max-num-seqs` above.
