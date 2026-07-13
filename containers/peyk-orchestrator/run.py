#!/usr/bin/env python3
"""Orchestrator CLI.

Usage:
    run.py --config <config.yaml> --input <dir> --output <dir> [--workdir <dir>]

Dispatches peyk-layout -> born-digital check -> peyk-dcr/peyk-ocr + peyk-tsr + peyk-vlm
per document, then assembles one markdown file per document using peyk-layout's own
region order — unless config.fullpage is set, which bypasses all of that for one
whole-page-at-a-time model instead. See pipeline.py and config/example.yaml.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

from config import load_config
from pipeline import _vlm_credential_docker_args, dispatch_documents, render_pdf_pages, run_layout
from stages import run_docker_stage

# Fixed convention rather than a tempfile.TemporaryDirectory(): /hotstorage is bind-mounted to
# one stable host directory both locally and on EC2 (see run_local.sh), so intermediate
# crops/renders are inspectable between runs instead of vanishing on process exit. A "workdir"
# subdir, not /hotstorage itself, since /hotstorage also holds the sibling input/ and output/
# dirs run_local.sh mounts alongside it. Still ephemeral, still local-disk-only — never S3,
# cleared per job as needed; overridable via --workdir for e.g. test harnesses.
DEFAULT_WORKDIR = Path("/hotstorage/workdir")


def _run_fullpage(config, args: argparse.Namespace) -> int:
    """config.fullpage's model determines the mechanics: "surya" renders PDFs itself (peyk-surya
    already has this built in, --mode fullpage) and bypasses run_layout/dispatch_documents (the
    normal per-region path) entirely — one docker run for the whole batch, using its output
    directly as final per-document markdown. Any other model key is a peyk-vlm model, which has
    no PDF-rendering capability of its own (no pypdfium2/PDF deps at all — it only ever takes
    already-rendered image files) — pages are rendered directly via render_pdf_pages() (plain
    pypdfium2 rasterization, no model) instead of running the full peyk-layout container just to
    get its raw-render side effect, since this mode has no use for peyk-layout's actual
    region-detection inference at all."""
    model = config.fullpage.backend
    print(f"[peyk-orchestrator] fullpage job, model={model!r}: ", end="", file=sys.stderr)

    if model == "surya":
        print("running peyk-surya over input batch (it renders PDFs itself)...", file=sys.stderr)
        # Dispatched into a workdir-internal scratch dir, not args.output directly:
        # run_docker_stage unconditionally rmtree's whatever output_dir it's given (safe for
        # every other caller, which always passes a workdir subdir — see stages.py) — args.output
        # is the persistent, host-bind-mounted directory the user actually reads results from
        # (see run_local.sh), so wiping it outright would destroy any prior run's output still
        # sitting there. Copying the produced .md files in below instead only touches the files
        # this run actually produced.
        surya_out = args.workdir / "fullpage_out"
        run_docker_stage(
            image=config.fullpage.image,
            model="surya",
            input_dir=args.input,
            output_dir=surya_out,
            extra_args=["--mode", "fullpage"],
        )
        for md_path in surya_out.glob("*.md"):
            (args.output / md_path.name).write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"[peyk-orchestrator] wrote {args.output / md_path.name}", file=sys.stderr)
        return 0

    print("rendering pages directly, then peyk-vlm over every page...", file=sys.stderr)
    docs = sorted(p for p in args.input.iterdir() if p.suffix.lower() == ".pdf")
    fullpage_render_dir = args.workdir / "fullpage_render"
    if fullpage_render_dir.exists():
        shutil.rmtree(fullpage_render_dir)
    fullpage_render_dir.mkdir(parents=True, exist_ok=True)
    fullpage_in = args.workdir / "fullpage_in"
    if fullpage_in.exists():
        shutil.rmtree(fullpage_in)
    fullpage_in.mkdir(parents=True, exist_ok=True)
    pages_by_doc: dict[str, dict[int, Path]] = {}
    for doc_path in docs:
        doc_stem = doc_path.stem
        pages = render_pdf_pages(doc_path, fullpage_render_dir)
        pages_by_doc[doc_stem] = pages
        for page_idx, page_path in pages.items():
            dest = fullpage_in / f"{doc_stem}__p{page_idx}.png"
            dest.write_bytes(page_path.read_bytes())
    fullpage_out = args.workdir / "fullpage_out"
    run_docker_stage(
        image=config.fullpage.image,
        model=model,
        input_dir=fullpage_in,
        output_dir=fullpage_out,
        extra_args=["--role", "fullpage"],
        extra_docker_args=_vlm_credential_docker_args(model),
        gpu=False,
    )
    page_results: dict[str, dict[int, str]] = {}
    for json_path in fullpage_out.glob("*.json"):
        doc_stem, page_str = json_path.stem.split("__p", 1)
        page_results.setdefault(doc_stem, {})[int(page_str)] = json.loads(json_path.read_text())["text"]
    for doc_stem, pages in pages_by_doc.items():
        doc_pages = page_results.get(doc_stem, {})
        markdown = "\n\n".join(doc_pages[page_idx] for page_idx in sorted(pages) if page_idx in doc_pages)
        out_path = args.output / f"{doc_stem}.md"
        out_path.write_text(markdown, encoding="utf-8")
        print(f"[peyk-orchestrator] wrote {out_path}", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Dispatch the peyk pipeline stages and assemble markdown output.")
    parser.add_argument("--config", required=True, type=Path, help="Path to pipeline config YAML.")
    parser.add_argument("--input", required=True, type=Path, help="Directory of input PDFs.")
    parser.add_argument("--output", required=True, type=Path, help="Directory to write assembled markdown, one file per document.")
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR, help=f"Directory for intermediate crops/renders (default: {DEFAULT_WORKDIR}, bind-mount this to a host directory).")
    args = parser.parse_args()

    if not args.input.is_dir():
        parser.error(f"--input {args.input} is not a directory")
    args.output.mkdir(parents=True, exist_ok=True)
    args.workdir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)

    if config.fullpage is not None:
        return _run_fullpage(config, args)

    print("[peyk-orchestrator] running peyk-layout over input batch...", file=sys.stderr)
    layout_results = run_layout(config, args.input, args.workdir)
    if not layout_results:
        print(f"[peyk-orchestrator] no layout output produced for {args.input}", file=sys.stderr)
        return 1

    docs = sorted(p for p in args.input.iterdir() if p.suffix.lower() == ".pdf")
    print(f"[peyk-orchestrator] dispatching {len(docs)} document(s)...", file=sys.stderr)
    markdowns = dispatch_documents(docs, layout_results, config, args.workdir)
    for doc_path in docs:
        markdown = markdowns.get(doc_path.stem)
        if markdown is None:
            print(f"[peyk-orchestrator] no layout regions for {doc_path.name}, skipping", file=sys.stderr)
            continue
        out_path = args.output / f"{doc_path.stem}.md"
        out_path.write_text(markdown, encoding="utf-8")
        print(f"[peyk-orchestrator] wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
