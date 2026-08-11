#!/usr/bin/env python3
"""Table Structure Recognition CLI.

Usage:
    run.py --model <backend> --input <dir> --output <dir> [--visualize] [--watch]

For each table-region crop image in --input, runs the selected TSR backend and
writes a "<crop-stem>.json" file to --output containing the recognized cell
grid (row/col count + per-cell bbox/span, in coordinates local to the crop).
Structure only, no text — peyk-orchestrator pairs each cell with peyk-dcr
(born-digital) or the OCR containers (scanned) itself; see pipeline.md /
implementation_plan.md Task 1.5.

Also writes a "<crop-stem>_aug.json" with: one row-band bbox per row (full
crop width, height calibrated from the model's cells), one column-band bbox
per column (full crop height, width calibrated from the model's cells), and
one regularized bbox per cell (row-band × column-band intersection, replacing
the model's raw per-cell box). All three are a safety net for downstream text
pairing/cropping when individual cell boxes clip or miss real content (e.g. a
header row whose per-cell boxes came out too tight/incomplete) — see
implementation_plan.md Task 1.5's row/column-based OCR+DCR follow-up.

With --visualize, also writes "<crop-stem>_viz.png" (raw per-cell boxes) and
"<crop-stem>_aug_viz.png" (row bands in orange, column bands in blue,
regularized cell grid in green) per crop to --output, for eyeballing
structure quality.

With --watch, the backend is loaded once and the process stays alive, polling
--input for new crops (any image without a matching output JSON yet) instead
of processing one batch and exiting.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from backends import BACKENDS
from backends.base import col_boxes, regularized_cells, row_boxes

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def draw_cells(image_path: Path, structure, out_path: Path) -> None:
    from PIL import Image, ImageDraw

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for cell in structure.cells:
        x0, y0, x1, y1 = cell.bbox
        draw.rectangle([x0, y0, x1, y1], outline=(0, 200, 0), width=2)
        draw.text((x0 + 2, max(0, y0 - 12)), f"r{cell.row}c{cell.col}", fill=(220, 0, 0))
    image.save(out_path)


def draw_aug(image_path: Path, rows, cols, cells: list[dict], out_path: Path) -> None:
    from PIL import Image, ImageDraw

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for row in rows:
        x0, y0, x1, y1 = row.bbox
        draw.rectangle([x0, y0, x1, y1], outline=(220, 130, 0), width=1)
    for col in cols:
        x0, y0, x1, y1 = col.bbox
        draw.rectangle([x0, y0, x1, y1], outline=(0, 100, 220), width=1)
    for cell in cells:
        x0, y0, x1, y1 = cell["bbox"]
        draw.rectangle([x0, y0, x1, y1], outline=(0, 200, 0), width=2)
    image.save(out_path)


def process_crop(crop_path: Path, backend, model: str, output_dir: Path, visualize: bool) -> None:
    print(f"[peyk-tsr] processing {crop_path.name}...", file=sys.stderr)
    structure = backend.predict(crop_path)
    out_path = output_dir / f"{crop_path.stem}.json"
    out_path.write_text(json.dumps({"crop": crop_path.name, "model": model, **structure.to_dict()}, indent=2))
    print(f"[peyk-tsr] wrote {out_path}", file=sys.stderr)

    from PIL import Image

    image = Image.open(crop_path)
    rows = row_boxes(structure, image.width)
    cols = col_boxes(structure, image.width, image.height)
    cells = regularized_cells(structure, rows, cols)
    aug_path = output_dir / f"{crop_path.stem}_aug.json"
    aug_path.write_text(json.dumps({
        "crop": crop_path.name,
        "model": model,
        "rows": [r.to_dict() for r in rows],
        "cols": [c.to_dict() for c in cols],
        "cells": cells,
    }, indent=2))
    print(f"[peyk-tsr] wrote {aug_path}", file=sys.stderr)

    if visualize:
        viz_path = output_dir / f"{crop_path.stem}_viz.png"
        draw_cells(crop_path, structure, viz_path)
        print(f"[peyk-tsr] wrote {viz_path}", file=sys.stderr)

        aug_viz_path = output_dir / f"{crop_path.stem}_aug_viz.png"
        draw_aug(crop_path, rows, cols, cells, aug_viz_path)
        print(f"[peyk-tsr] wrote {aug_viz_path}", file=sys.stderr)


def watch(input_dir: Path, output_dir: Path, backend, model: str, visualize: bool, poll_interval: float) -> int:
    print(f"[peyk-tsr] watching {input_dir} for new crops (Ctrl+C to stop)...", file=sys.stderr)
    try:
        while True:
            crops = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            pending = [p for p in crops if not (output_dir / f"{p.stem}.json").exists()]
            for crop_path in pending:
                process_crop(crop_path, backend, model, output_dir, visualize)
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("[peyk-tsr] stopped.", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run table structure recognition over a batch of table-region crops.")
    parser.add_argument("--model", required=True, choices=sorted(BACKENDS.keys()), help="TSR backend to use.")
    parser.add_argument("--input", required=True, type=Path, help="Directory of input table-region crop images.")
    parser.add_argument("--output", required=True, type=Path, help="Directory to write per-crop cell-grid JSON.")
    parser.add_argument("--visualize", action="store_true", help="Also write a per-crop PNG with detected cell boxes drawn on top, for testing.")
    parser.add_argument("--watch", action="store_true", help="Load the model once, then keep polling --input for new crops instead of exiting after one batch.")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Seconds between --watch polls (default: 1.0).")
    args = parser.parse_args()

    if not args.input.is_dir():
        parser.error(f"--input {args.input} is not a directory")
    args.output.mkdir(parents=True, exist_ok=True)

    backend_cls = BACKENDS[args.model]
    backend = backend_cls()
    print(f"[peyk-tsr] loading backend '{args.model}'...", file=sys.stderr)
    backend.load()
    print(f"[peyk-tsr] backend '{args.model}' loaded.", file=sys.stderr)

    if args.watch:
        return watch(args.input, args.output, backend, args.model, args.visualize, args.poll_interval)

    crops = sorted(p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not crops:
        print(f"[peyk-tsr] no crop images found in {args.input}", file=sys.stderr)
        return 1

    for crop_path in crops:
        process_crop(crop_path, backend, args.model, args.output, args.visualize)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
