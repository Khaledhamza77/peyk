#!/usr/bin/env python3
"""Orchestrator CLI.

Usage:
    run.py --config <config.yaml> --input <dir> --output <dir> [--workdir <dir>]

Dispatches peyk-layout -> born-digital check -> peyk-dcr/peyk-ocr + peyk-tsr + peyk-vlm
per document, then assembles one markdown file per document using peyk-layout's own
region order. Stages without a built image yet are stubbed via `stub: true` in the config
— see pipeline.py and config/example.yaml.
"""
import argparse
import sys
import tempfile
from pathlib import Path

from config import load_config
from pipeline import run_layout, dispatch_document


def main() -> int:
    parser = argparse.ArgumentParser(description="Dispatch the peyk pipeline stages and assemble markdown output.")
    parser.add_argument("--config", required=True, type=Path, help="Path to pipeline config YAML.")
    parser.add_argument("--input", required=True, type=Path, help="Directory of input PDFs.")
    parser.add_argument("--output", required=True, type=Path, help="Directory to write assembled markdown, one file per document.")
    parser.add_argument("--workdir", type=Path, default=None, help="Directory for intermediate crops/renders (default: a temp dir).")
    args = parser.parse_args()

    if not args.input.is_dir():
        parser.error(f"--input {args.input} is not a directory")
    args.output.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)

    def _run(workdir: Path) -> int:
        print("[peyk-orchestrator] running peyk-layout over input batch...", file=sys.stderr)
        layout_results = run_layout(config, args.input, workdir)
        if not layout_results:
            print(f"[peyk-orchestrator] no layout output produced for {args.input}", file=sys.stderr)
            return 1

        docs = sorted(p for p in args.input.iterdir() if p.suffix.lower() == ".pdf")
        for doc_path in docs:
            if doc_path.stem not in layout_results:
                print(f"[peyk-orchestrator] no layout regions for {doc_path.name}, skipping", file=sys.stderr)
                continue
            print(f"[peyk-orchestrator] dispatching {doc_path.name}...", file=sys.stderr)
            markdown = dispatch_document(doc_path, layout_results[doc_path.stem], config, workdir)
            out_path = args.output / f"{doc_path.stem}.md"
            out_path.write_text(markdown, encoding="utf-8")
            print(f"[peyk-orchestrator] wrote {out_path}", file=sys.stderr)
        return 0

    if args.workdir:
        args.workdir.mkdir(parents=True, exist_ok=True)
        return _run(args.workdir)
    with tempfile.TemporaryDirectory() as tmp:
        return _run(Path(tmp))


if __name__ == "__main__":
    raise SystemExit(main())
