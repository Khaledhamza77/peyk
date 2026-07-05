#!/usr/bin/env python3
"""Text OCR CLI.

Usage:
    run.py --model <backend> --input <dir> --output <dir>

For each text-region crop image in --input, runs the selected OCR backend and
writes a "<crop-stem>.json" file to --output containing the recognized text.
"""
import argparse
import json
import sys
from pathlib import Path

from backends import BACKENDS

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


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
    args = parser.parse_args()

    if not args.input.is_dir():
        parser.error(f"--input {args.input} is not a directory")
    args.output.mkdir(parents=True, exist_ok=True)

    backend_cls = BACKENDS[args.model]
    backend = backend_cls(lang=args.lang)
    print(f"[peyk-ocr] loading backend '{args.model}'...", file=sys.stderr)
    backend.load()

    crops = sorted(p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not crops:
        print(f"[peyk-ocr] no crop images found in {args.input}", file=sys.stderr)
        return 1

    for crop_path in crops:
        print(f"[peyk-ocr] processing {crop_path.name}...", file=sys.stderr)
        result = backend.predict(crop_path)

        out_path = args.output / f"{crop_path.stem}.json"
        out_path.write_text(json.dumps({"crop": crop_path.name, "model": args.model, **result.to_dict()}, indent=2))
        print(f"[peyk-ocr] wrote {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
