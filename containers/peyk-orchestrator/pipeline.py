"""Core dispatch: peyk-layout -> born-digital check -> per-region stage dispatch ->
assembly. Regions are handed to their stage in `peyk-layout`'s own output order and
reassembled in that same order (see pipeline.md — no separate reading-order solver)."""
import json
import shutil
import sys
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

from config import PipelineConfig
from stages import run_docker_stage, stub_fragment

RENDER_SCALE = 300 / 72  # must match peyk-layout's render scale so region bboxes line up


def render_pages(doc_path: Path, tmp_dir: Path) -> dict[int, Path]:
    pdf = pdfium.PdfDocument(str(doc_path))
    pages = {}
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            bitmap = page.render(scale=RENDER_SCALE)
            out = tmp_dir / f"{doc_path.stem}_p{i}.png"
            bitmap.to_pil().save(out)
            pages[i] = out
    finally:
        pdf.close()
    return pages


def born_digital_pages(doc_path: Path, min_chars: int, force_scanned: bool = False) -> dict[int, bool]:
    pdf = pdfium.PdfDocument(str(doc_path))
    result = {}
    try:
        for i in range(len(pdf)):
            if force_scanned:
                result[i] = False
                continue
            text = pdf[i].get_textpage().get_text_range()
            result[i] = len(text.strip()) >= min_chars
    finally:
        pdf.close()
    return result


# Layout detectors are trained mostly on Latin/CJK text density and often don't account
# for marks that extend beyond the visual line body — Arabic diacritics in particular
# (fatha/damma/kasra/shadda/sukun) can sit just outside a pixel-tight bbox and get sliced
# off before OCR ever sees them. A small fixed margin gives them room without pulling in
# much of a neighboring region.
CROP_PADDING_PX = 6


def crop_region(page_image_path: Path, bbox: list[float], out_path: Path) -> None:
    image = Image.open(page_image_path)
    x0, y0, x1, y1 = bbox
    x0 = max(0, int(x0) - CROP_PADDING_PX)
    y0 = max(0, int(y0) - CROP_PADDING_PX)
    x1 = min(image.width, int(x1) + CROP_PADDING_PX)
    y1 = min(image.height, int(y1) + CROP_PADDING_PX)
    image.crop((x0, y0, x1, y1)).save(out_path)


def run_layout(config: PipelineConfig, input_dir: Path, workdir: Path) -> dict[str, dict]:
    """Batch layout over every document in input_dir. Returns {doc_stem: regions_json}."""
    out_dir = workdir / "layout_out"
    run_docker_stage(
        image=config.layout.image,
        model=config.layout.backend,
        input_dir=input_dir,
        output_dir=out_dir,
        extra_args=["--visualize"],
    )
    return {p.stem: json.loads(p.read_text()) for p in out_dir.glob("*.json")}


def dispatch_document(doc_path: Path, regions_doc: dict, config: PipelineConfig, workdir: Path) -> str:
    """Crop every region, dispatch each to its stage, assemble in layout order."""
    doc_stem = doc_path.stem
    render_dir = workdir / "pages" / doc_stem
    render_dir.mkdir(parents=True, exist_ok=True)
    pages = render_pages(doc_path, render_dir)

    born_digital = born_digital_pages(doc_path, config.born_digital_min_chars, config.force_scanned)

    regions = regions_doc["regions"]
    crops_dir = workdir / "crops" / doc_stem
    # Cleared, not just mkdir(exist_ok=True): a stale crop left over from a previous run at
    # this same path would otherwise get fed into a stage as if it belonged to this run.
    if crops_dir.exists():
        shutil.rmtree(crops_dir)
    crops_dir.mkdir(parents=True, exist_ok=True)

    ocr_targets = []  # (region_idx, crop_path) — scanned text regions needing OCR
    for idx, region in enumerate(regions):
        page_image = pages[region["page"]]
        crop_path = crops_dir / f"r{idx}.png"
        crop_region(page_image, region["bbox"], crop_path)
        region["_crop"] = crop_path
        region["_born_digital"] = born_digital.get(region["page"], False)
        # Tables never go to peyk-ocr as a whole-region image — feeding a full table crop
        # into a line-oriented OCR call produced garbage/repeated-token output in testing.
        # peyk-tsr (once real) does the scanned-table + OCR pairing itself, per pipeline.md,
        # by running OCR per detected cell rather than over the whole table image.
        if region["label"] == "text" and not region["_born_digital"]:
            ocr_targets.append((idx, crop_path))

    ocr_results: dict[int, str] = {}
    if ocr_targets and not config.ocr.stub:
        ocr_in = workdir / "ocr_in" / doc_stem
        if ocr_in.exists():
            shutil.rmtree(ocr_in)
        ocr_in.mkdir(parents=True, exist_ok=True)
        idx_by_stem = {}
        for idx, crop_path in ocr_targets:
            dest = ocr_in / f"r{idx}{crop_path.suffix}"
            dest.write_bytes(crop_path.read_bytes())
            idx_by_stem[dest.stem] = idx
        ocr_extra_args = ["--lang", config.ocr.lang]
        if config.ocr.server_url:
            ocr_extra_args += ["--server-url", config.ocr.server_url]
        ocr_out = workdir / "ocr_out" / doc_stem
        run_docker_stage(
            image=config.ocr.image,
            model=config.ocr.backend,
            input_dir=ocr_in,
            output_dir=ocr_out,
            extra_args=ocr_extra_args,
        )
        for json_path in ocr_out.glob("*.json"):
            ocr_results[idx_by_stem[json_path.stem]] = json.loads(json_path.read_text())["text"]

    fragments = []
    for idx, region in enumerate(regions):
        label = region["label"]
        if label == "text":
            if region["_born_digital"]:
                fragments.append(stub_fragment("peyk-dcr", "text"))
            else:
                fragments.append(ocr_results.get(idx, stub_fragment("peyk-ocr", "text")))
        elif label == "table":
            fragments.append(stub_fragment("peyk-tsr", "table"))
        elif label == "figure":
            fragments.append(stub_fragment("peyk-vlm", "figure"))
        else:
            print(f"[peyk-orchestrator] unknown region label '{label}', skipping", file=sys.stderr)

    return "\n\n".join(fragments)
