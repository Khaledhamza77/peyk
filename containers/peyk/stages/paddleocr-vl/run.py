#!/usr/bin/env python3
"""PaddleOCR-VL OCR CLI — a thin client against a persistent vLLM server.

Usage:
    run.py --input <dir> --output <dir> --server-url <url> [--concurrency N] [--watch]

For each text-region crop image in --input, sends it to the PaddleOCR-VL vLLM
server at --server-url and writes a "<crop-stem>.json" file to --output
containing the recognized text. Unlike peyk-simple-ocr, this container does no
local model inference and needs no GPU of its own — see backends/paddleocr_vl.py
and peyk-vllm-paddleocr/ for why the model moved out of this process.

Crops are dispatched concurrently (--concurrency, default 8), not one at a time:
vLLM's whole performance story is continuous batching, which only kicks in when
multiple requests are in flight at once. A prior version of this file looped
over crops sequentially — one blocking HTTP round-trip per crop, GPU idling
between requests — so a table with 100+ cells paid full per-request latency
100+ times over with none of vLLM's batching ever engaged. Threads (not
asyncio): backend.predict() is a blocking call through PaddleX's genai_client/
openai HTTP client, and each call is I/O-bound waiting on the server, so the
GIL is released during the actual wait and threads give real concurrency here.

With --watch, the backend is loaded once and the process stays alive, polling
--input for new crops (any image without a matching output JSON yet) instead
of processing one batch and exiting — for keeping the client warm across
repeated manual/dev test requests instead of paying the load cost every call.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from backends import BACKENDS

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Matches peyk-vllm-paddleocr's default port (see that container's README). Overridable via
# --server-url for pointing at a differently-hosted server.
DEFAULT_SERVER_URL = "http://peyk-vllm-paddleocr:8118/v1"

# Raised from the original starting point of 8: a single table can have 100+ cells, each one
# its own generation call, and 8 in flight at once meant 12+ sequential rounds before vLLM's
# continuous batching had any real headroom to exploit. vLLM's default max_num_seqs (not
# overridden in peyk-vllm-paddleocr/vllm_config.yml) is generous enough that this is still
# well under what the server can schedule concurrently — not independently load-tested at
# this exact value, so tune down if requests start queuing/timing out, or reconsider
# vllm_config.yml's own gpu_memory_utilization/max_num_seqs if the server turns out to be the
# actual bottleneck rather than this client-side cap.
DEFAULT_CONCURRENCY = 32


def process_crop(crop_path: Path, backend, model: str, output_dir: Path) -> None:
    print(f"[peyk-paddleocr-vl] processing {crop_path.name}...", file=sys.stderr)
    result = backend.predict(crop_path)
    out_path = output_dir / f"{crop_path.stem}.json"
    out_path.write_text(json.dumps({"crop": crop_path.name, "model": model, **result.to_dict()}, indent=2, ensure_ascii=False))
    print(f"[peyk-paddleocr-vl] wrote {out_path}", file=sys.stderr)


def process_batch(crops: list[Path], backend, model: str, output_dir: Path, concurrency: int) -> None:
    # Sharing one `backend` (and its underlying HTTP client) across worker threads rather
    # than one per thread — PaddleX's genai_client wraps an openai-SDK-style client, which is
    # documented thread-safe for concurrent requests; not independently re-verified here, but
    # each call only reads shared state (server_url) and writes to its own output path, so a
    # data race in our own code isn't possible even if the underlying client serializes
    # internally.
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(process_crop, crop_path, backend, model, output_dir): crop_path for crop_path in crops}
        for future in as_completed(futures):
            crop_path = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[peyk-paddleocr-vl] {crop_path.name} failed: {e}", file=sys.stderr)


def watch(input_dir: Path, output_dir: Path, backend, model: str, poll_interval: float, concurrency: int) -> int:
    print(f"[peyk-paddleocr-vl] watching {input_dir} for new crops (Ctrl+C to stop)...", file=sys.stderr)
    try:
        while True:
            crops = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            pending = [p for p in crops if not (output_dir / f"{p.stem}.json").exists()]
            if pending:
                process_batch(pending, backend, model, output_dir, concurrency)
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("[peyk-paddleocr-vl] stopped.", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PaddleOCR-VL (via a vLLM server) over a batch of text-region crops.")
    parser.add_argument("--model", default="paddleocr-vl", choices=sorted(BACKENDS.keys()), help="OCR backend to use.")
    parser.add_argument(
        "--lang",
        default="arabic",
        choices=["arabic", "latin"],
        help="Script of the crops being processed (unused by this backend; accepted for CLI-interface parity with peyk-simple-ocr).",
    )
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL, help=f"Base URL of the PaddleOCR-VL vLLM server (default: {DEFAULT_SERVER_URL}).")
    parser.add_argument("--input", required=True, type=Path, help="Directory of input crop images.")
    parser.add_argument("--output", required=True, type=Path, help="Directory to write per-crop recognized text.")
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help=f"Max in-flight requests to the vLLM server at once (default: {DEFAULT_CONCURRENCY}) — lets vLLM's "
             "continuous batching actually batch across cells instead of processing them one at a time.",
    )
    parser.add_argument("--watch", action="store_true", help="Load the model once, then keep polling --input for new crops instead of exiting after one batch.")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Seconds between --watch polls (default: 1.0).")
    args = parser.parse_args()

    if not args.input.is_dir():
        parser.error(f"--input {args.input} is not a directory")
    args.output.mkdir(parents=True, exist_ok=True)

    backend_cls = BACKENDS[args.model]
    backend = backend_cls(lang=args.lang, server_url=args.server_url)
    print(f"[peyk-paddleocr-vl] loading backend '{args.model}' (server: {args.server_url})...", file=sys.stderr)
    backend.load()
    print(f"[peyk-paddleocr-vl] backend '{args.model}' loaded.", file=sys.stderr)

    if args.watch:
        return watch(args.input, args.output, backend, args.model, args.poll_interval, args.concurrency)

    crops = sorted(p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not crops:
        print(f"[peyk-paddleocr-vl] no crop images found in {args.input}", file=sys.stderr)
        return 1

    process_batch(crops, backend, args.model, args.output, args.concurrency)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
