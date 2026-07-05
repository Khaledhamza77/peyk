#!/usr/bin/env python3
"""Layout detection CLI.

Usage:
    run.py --model <backend> --input <dir> --output <dir>

For each document in --input (PDF or image), runs the selected layout backend
over every page and writes a "<doc-stem>.json" file to --output containing the
detected regions (bbox + class label) per page.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

from backends import BACKENDS

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def iter_page_images(doc_path: Path, tmp_dir: Path):
    """Yield (page_index, image_path) for every page of doc_path."""
    if doc_path.suffix.lower() == ".pdf":
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(doc_path))
        try:
            for page_index in range(len(pdf)):
                page = pdf[page_index]
                bitmap = page.render(scale=200 / 72)
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run layout detection over a batch of documents.")
    parser.add_argument("--model", required=True, choices=sorted(BACKENDS.keys()), help="Layout backend to use.")
    parser.add_argument("--input", required=True, type=Path, help="Directory of input PDFs/images.")
    parser.add_argument("--output", required=True, type=Path, help="Directory to write per-document region JSON.")
    args = parser.parse_args()

    if not args.input.is_dir():
        parser.error(f"--input {args.input} is not a directory")
    args.output.mkdir(parents=True, exist_ok=True)

    backend_cls = BACKENDS[args.model]
    backend = backend_cls()
    print(f"[peyk-layout] loading backend '{args.model}'...", file=sys.stderr)
    backend.load()

    docs = sorted(
        p for p in args.input.iterdir() if p.suffix.lower() == ".pdf" or p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not docs:
        print(f"[peyk-layout] no PDF/image files found in {args.input}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for doc_path in docs:
            print(f"[peyk-layout] processing {doc_path.name}...", file=sys.stderr)
            pages_out = []
            for page_index, image_path in iter_page_images(doc_path, tmp_dir):
                regions = backend.predict(image_path)
                for region in regions:
                    region.page = page_index
                pages_out.extend(r.to_dict() for r in regions)

            out_path = args.output / f"{doc_path.stem}.json"
            out_path.write_text(json.dumps({"document": doc_path.name, "model": args.model, "regions": pages_out}, indent=2))
            print(f"[peyk-layout] wrote {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
