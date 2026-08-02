#!/usr/bin/env python3
"""Text OCR CLI.

Usage:
    run.py --model <backend> --input <dir> --output <dir> [--watch]

For each text-region crop image in --input, runs the selected OCR backend and
writes a "<crop-stem>.json" file to --output containing the recognized text.

With --watch, the backend is loaded once and the process stays alive, polling
--input for new crops (any image without a matching output JSON yet) instead
of processing one batch and exiting — for keeping the model warm across
repeated manual/dev test requests instead of paying the load cost every call.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from backends import BACKENDS

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def process_crop(crop_path: Path, backend, model: str, output_dir: Path) -> None:
    print(f"[peyk-simple-ocr] processing {crop_path.name}...", file=sys.stderr)
    result = backend.predict(crop_path)
    out_path = output_dir / f"{crop_path.stem}.json"
    out_path.write_text(json.dumps({"crop": crop_path.name, "model": model, **result.to_dict()}, indent=2, ensure_ascii=False))
    print(f"[peyk-simple-ocr] wrote {out_path}", file=sys.stderr)


def watch(input_dir: Path, output_dir: Path, backend, model: str, poll_interval: float) -> int:
    print(f"[peyk-simple-ocr] watching {input_dir} for new crops (Ctrl+C to stop)...", file=sys.stderr)
    try:
        while True:
            crops = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            pending = [p for p in crops if not (output_dir / f"{p.stem}.json").exists()]
            for crop_path in pending:
                process_crop(crop_path, backend, model, output_dir)
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("[peyk-simple-ocr] stopped.", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OCR over a batch of text-region crops.")
    parser.add_argument("--model", required=True, choices=sorted(BACKENDS.keys()), help="OCR backend to use.")
    parser.add_argument(
        "--lang",
        default="arabic",
        choices=["arabic", "latin"],
        help="Script of the crops being processed (default: arabic — most company documents are Arabic).",
    )
    parser.add_argument("--input", required=True, type=Path, help="Directory of input crop images.")
    parser.add_argument("--output", required=True, type=Path, help="Directory to write per-crop recognized text.")
    parser.add_argument("--watch", action="store_true", help="Load the model once, then keep polling --input for new crops instead of exiting after one batch.")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Seconds between --watch polls (default: 1.0).")
    args = parser.parse_args()

    if not args.input.is_dir():
        parser.error(f"--input {args.input} is not a directory")
    args.output.mkdir(parents=True, exist_ok=True)

    backend_cls = BACKENDS[args.model]
    backend = backend_cls(lang=args.lang)
    print(f"[peyk-simple-ocr] loading backend '{args.model}'...", file=sys.stderr)
    backend.load()
    print(f"[peyk-simple-ocr] backend '{args.model}' loaded.", file=sys.stderr)

    if args.watch:
        return watch(args.input, args.output, backend, args.model, args.poll_interval)

    crops = sorted(p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not crops:
        print(f"[peyk-simple-ocr] no crop images found in {args.input}", file=sys.stderr)
        return 1

    for crop_path in crops:
        process_crop(crop_path, backend, args.model, args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
