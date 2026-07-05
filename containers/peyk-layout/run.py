#!/usr/bin/env python3
"""Layout detection CLI.

Usage:
    run.py --model <backend> --input <dir> --output <dir> [--visualize] [--watch]

For each document in --input (PDF or image), runs the selected layout backend
over every page and writes a "<doc-stem>.json" file to --output containing the
detected regions (bbox + class label) per page. With --visualize, also writes a
"<doc-stem>_p<page>.png" per page to --output with detected boxes drawn on top
of the rendered page, for eyeballing detection quality.

With --watch, the backend is loaded once and the process stays alive, polling
--input for new documents (any doc without a matching output JSON yet) instead
of processing one batch and exiting — for keeping the model warm across
repeated manual/dev test requests instead of paying the load cost every call.
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from backends import BACKENDS

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# 300 DPI rather than 200 — more pixels per glyph helps OCR recognition accuracy on bold/
# small text, at some cost to layout inference time. Must match peyk-orchestrator's
# pipeline.py RENDER_SCALE exactly, since region bboxes are computed at this scale and
# reused there to crop the same page rendered again independently.
RENDER_SCALE = 300 / 72

_LABEL_COLORS = {
    "text": (0, 128, 255),
    "table": (0, 200, 0),
    "figure": (220, 0, 0),
}


def draw_regions(image_path: Path, regions, out_path: Path) -> None:
    from PIL import Image, ImageDraw

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for region in regions:
        color = _LABEL_COLORS.get(region.label, (255, 165, 0))
        x0, y0, x1, y1 = region.bbox
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        draw.text((x0 + 2, max(0, y0 - 12)), f"{region.label} {region.score:.2f}", fill=color)
    image.save(out_path)


def iter_page_images(doc_path: Path, tmp_dir: Path):
    """Yield (page_index, image_path) for every page of doc_path."""
    if doc_path.suffix.lower() == ".pdf":
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(doc_path))
        try:
            for page_index in range(len(pdf)):
                page = pdf[page_index]
                bitmap = page.render(scale=RENDER_SCALE)
                image = bitmap.to_pil()
                out_path = tmp_dir / f"{doc_path.stem}_p{page_index}.png"
                image.save(out_path)
                yield page_index, out_path
        finally:
            pdf.close()
    elif doc_path.suffix.lower() in IMAGE_SUFFIXES:
        yield 0, doc_path
    else:
        raise ValueError(f"Unsupported input file type: {doc_path.suffix}")


def process_doc(doc_path: Path, backend, model: str, output_dir: Path, tmp_dir: Path, visualize: bool) -> None:
    print(f"[peyk-layout] processing {doc_path.name}...", file=sys.stderr)
    pages_out = []
    for page_index, image_path in iter_page_images(doc_path, tmp_dir):
        regions = backend.predict(image_path)
        for region in regions:
            region.page = page_index
        pages_out.extend(r.to_dict() for r in regions)
        if visualize:
            viz_path = output_dir / f"{doc_path.stem}_p{page_index}.png"
            draw_regions(image_path, regions, viz_path)

    out_path = output_dir / f"{doc_path.stem}.json"
    out_path.write_text(json.dumps({"document": doc_path.name, "model": model, "regions": pages_out}, indent=2, ensure_ascii=False))
    print(f"[peyk-layout] wrote {out_path}", file=sys.stderr)


def watch(input_dir: Path, output_dir: Path, backend, model: str, visualize: bool, poll_interval: float) -> int:
    print(f"[peyk-layout] watching {input_dir} for new documents (Ctrl+C to stop)...", file=sys.stderr)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            while True:
                docs = sorted(
                    p for p in input_dir.iterdir() if p.suffix.lower() == ".pdf" or p.suffix.lower() in IMAGE_SUFFIXES
                )
                pending = [p for p in docs if not (output_dir / f"{p.stem}.json").exists()]
                for doc_path in pending:
                    process_doc(doc_path, backend, model, output_dir, tmp_dir, visualize)
                time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("[peyk-layout] stopped.", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run layout detection over a batch of documents.")
    parser.add_argument("--model", required=True, choices=sorted(BACKENDS.keys()), help="Layout backend to use.")
    parser.add_argument("--input", required=True, type=Path, help="Directory of input PDFs/images.")
    parser.add_argument("--output", required=True, type=Path, help="Directory to write per-document region JSON.")
    parser.add_argument("--visualize", action="store_true", help="Also write a per-page PNG with detected boxes drawn on top, for testing.")
    parser.add_argument("--watch", action="store_true", help="Load the model once, then keep polling --input for new documents instead of exiting after one batch.")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Seconds between --watch polls (default: 1.0).")
    args = parser.parse_args()

    if not args.input.is_dir():
        parser.error(f"--input {args.input} is not a directory")
    args.output.mkdir(parents=True, exist_ok=True)

    backend_cls = BACKENDS[args.model]
    backend = backend_cls()
    print(f"[peyk-layout] loading backend '{args.model}'...", file=sys.stderr)
    backend.load()
    print(f"[peyk-layout] backend '{args.model}' loaded.", file=sys.stderr)

    if args.watch:
        return watch(args.input, args.output, backend, args.model, args.visualize, args.poll_interval)

    docs = sorted(
        p for p in args.input.iterdir() if p.suffix.lower() == ".pdf" or p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not docs:
        print(f"[peyk-layout] no PDF/image files found in {args.input}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for doc_path in docs:
            process_doc(doc_path, backend, args.model, args.output, tmp_dir, args.visualize)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
