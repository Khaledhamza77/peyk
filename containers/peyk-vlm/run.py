#!/usr/bin/env python3
"""peyk-vlm CLI — a thin client against managed cloud LLM APIs (Bedrock, Vertex AI).

Usage:
    run.py --model <key> --role {ocr,figure,table,fullpage} --input <dir> --output <dir>
           [--concurrency N] [--watch]

For each image in --input, sends it (with a role-specific prompt, see prompts.py) to the
selected model and writes a "<image-stem>.json" file to --output containing
{"crop", "model", "role", "text", "score"}. No local model/GPU — every --model is a remote API
call (see backends/registry.py for the cookie-cutter model list), so images are dispatched
concurrently (--concurrency, default 8) rather than one at a time, same rationale as
peyk-paddleocr-vl/peyk-surya: each call is I/O-bound waiting on the remote API, and threads
(not asyncio) give real concurrency since boto3/openai/google-genai's HTTP clients release the
GIL during the wait.

With --watch, the backend is loaded once and the process stays alive, polling --input for new
images (any image without a matching output JSON yet) instead of processing one batch and
exiting.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from backends import MODEL_REGISTRY, get_backend

# Anything this container's identity is validated against — peyk-orchestrator's config.py
# queries this directly (`--list-models`) rather than guessing which models are peyk-vlm's
# from a naming convention (e.g. assuming every key starts with "bedrock-"/"vertex-") — this
# is the one place that convention could silently drift out of sync with MODEL_REGISTRY.
# Printed as "<model-key>\t<provider>" so the orchestrator can also look up which cloud a
# model's credentials need, instead of guessing that from the key's spelling too.

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

ROLES = ["ocr", "figure", "table", "fullpage"]

DEFAULT_CONCURRENCY = 8


def process_image(image_path: Path, backend, model: str, role: str, output_dir: Path) -> None:
    print(f"[peyk-vlm] processing {image_path.name} (role={role})...", file=sys.stderr)
    result = backend.predict(image_path, role)
    out_path = output_dir / f"{image_path.stem}.json"
    out_path.write_text(
        json.dumps({"crop": image_path.name, "model": model, "role": role, **result.to_dict()}, indent=2, ensure_ascii=False)
    )
    print(f"[peyk-vlm] wrote {out_path}", file=sys.stderr)


def process_batch(images: list[Path], backend, model: str, role: str, output_dir: Path, concurrency: int) -> None:
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(process_image, image_path, backend, model, role, output_dir): image_path for image_path in images}
        for future in as_completed(futures):
            image_path = futures[future]
            try:
                future.result()
            except Exception as e:
                # Deliberately loud, not swallowed into an empty/garbage result written to
                # disk — a silent failure here is exactly the Surya RecognitionPredictor bug
                # (build_notes.md Task 1.8) this project already treated as a real bug.
                print(f"[peyk-vlm] {image_path.name} failed: {e}", file=sys.stderr)


def watch(input_dir: Path, output_dir: Path, backend, model: str, role: str, poll_interval: float, concurrency: int) -> int:
    print(f"[peyk-vlm] watching {input_dir} for new images (Ctrl+C to stop)...", file=sys.stderr)
    try:
        while True:
            images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            pending = [p for p in images if not (output_dir / f"{p.stem}.json").exists()]
            if pending:
                process_batch(pending, backend, model, role, output_dir, concurrency)
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("[peyk-vlm] stopped.", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a managed cloud LLM over a batch of images for a given role.")
    parser.add_argument(
        "--list-models", action="store_true",
        help="Print '<model-key>\\t<provider>' for every registered model, one per line, and exit — "
             "the ground truth peyk-orchestrator's config.py validates model names against, instead "
             "of assuming a naming convention.",
    )
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY.keys()), help="Model to invoke (see backends/registry.py).")
    parser.add_argument("--role", choices=ROLES, help="Which prompt/task to run: ocr | figure | table | fullpage.")
    parser.add_argument("--input", type=Path, help="Directory of input images (crops, or whole pages for --role fullpage).")
    parser.add_argument("--output", type=Path, help="Directory to write per-image results.")
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help=f"Max in-flight requests to the remote API at once (default: {DEFAULT_CONCURRENCY}).",
    )
    parser.add_argument("--watch", action="store_true", help="Load the backend once, then keep polling --input for new images instead of exiting after one batch.")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Seconds between --watch polls (default: 1.0).")
    args = parser.parse_args()

    if args.list_models:
        for key in sorted(MODEL_REGISTRY.keys()):
            print(f"{key}\t{MODEL_REGISTRY[key]['provider']}")
        return 0

    if args.model is None or args.role is None or args.input is None or args.output is None:
        parser.error("--model, --role, --input, and --output are all required unless --list-models is given")

    if not args.input.is_dir():
        parser.error(f"--input {args.input} is not a directory")
    args.output.mkdir(parents=True, exist_ok=True)

    backend = get_backend(args.model)
    print(f"[peyk-vlm] loading backend '{args.model}'...", file=sys.stderr)
    backend.load()
    print(f"[peyk-vlm] backend '{args.model}' loaded.", file=sys.stderr)

    if args.watch:
        return watch(args.input, args.output, backend, args.model, args.role, args.poll_interval, args.concurrency)

    images = sorted(p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        print(f"[peyk-vlm] no images found in {args.input}", file=sys.stderr)
        return 1

    process_batch(images, backend, args.model, args.role, args.output, args.concurrency)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
