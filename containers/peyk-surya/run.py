#!/usr/bin/env python3
"""Surya-OCR-2 CLI — a thin client against a persistent peyk-vllm-surya vLLM server (see that
container's README). Two independent modes:

Stage mode (default) — matches one of peyk-layout/peyk-tsr/peyk-simple-ocr's exact per-stage
output contracts, selected via --stage, so peyk-orchestrator can swap "surya" in as the
backend for layout.backend/tsr.backend/ocr.backend one at a time, same as any other backend.
table-full is the exception: not a drop-in replacement for another container's stage — it's
peyk-orchestrator's dedicated dispatch for the tsr.backend=="surya" case when the backend that
would actually OCR table cells (cell_ocr if set, else ocr) is also "surya" (config.py's
full_surya_tables — deliberately independent of ocr.backend itself, which only governs
non-table text), using predict_full (structure+text in one call per table) instead of the
usual structure-then-per-cell-OCR split every other backend combination goes through (see
pipeline.py and implementation_plan.md Task 1.5/1.8 for why isolated per-cell OCR is avoided
when possible):

    run.py --mode stage --stage layout      --input <dir> --output <dir> [--visualize]
    run.py --mode stage --stage tsr         --input <dir> --output <dir> [--visualize]
    run.py --mode stage --stage ocr         --input <dir> --output <dir>
    run.py --mode stage --stage table-full  --input <dir> --output <dir>

Fullpage mode — one warm in-process run per document, zero layout inference, one
RecognitionPredictor call per page. This used to offer four selectable shapes
(docs/surya/high_level.md); three were dropped after review concluded they didn't offer
anything worth a separate config choice — see run_fullpage's docstring for the reasoning per
option. What's left needs no numbered choice anymore:

    run.py --mode fullpage --input <dir> --output <dir>

See implementation_plan.md Task 1.8 for the design this is built from, and backends/client.py
for the "unconfirmed API shape" caveat that still applies to layout/tsr/table-full's predictor
calls (ocr/--stage ocr's RecognitionPredictor shape is confirmed; see
_blocks_from_recognition_result's docstring).
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from backends.base import Region, TableStructure, Cell, OCRResult, row_boxes, col_boxes, regularized_cells
from backends.client import SuryaClient, LABEL_MAP

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Must match peyk-layout's RENDER_SCALE / peyk-orchestrator's pipeline.py RENDER_SCALE exactly
# — region bboxes are computed at this scale and reused elsewhere to crop the same page
# rendered again independently. See peyk-layout/run.py's RENDER_SCALE comment for the full
# rationale (300 DPI, not 200, for glyph-level OCR accuracy).
RENDER_SCALE = 300 / 72

DEFAULT_SERVER_URL = "http://peyk-vllm-surya:8000/v1"


def iter_page_images(doc_path: Path, tmp_dir: Path):
    """Yield (page_index, image_path) for every page of doc_path. Duplicated from
    peyk-layout/run.py rather than imported — see backends/base.py's module docstring for why
    duplication across containers is the deliberate choice here."""
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


# ---------------------------------------------------------------------------
# --stage layout
# ---------------------------------------------------------------------------


def _regions_from_layout_result(layout_result) -> list[Region]:
    """Maps Surya's raw layout result onto this project's Region dataclass. See
    backends/client.py's module docstring: the exact attribute names here (`.bboxes`,
    `.label`, `.confidence`, `.bbox`) are the general Surya convention, not yet confirmed
    against a live prediction from this specific model/server."""
    regions = []
    for box in layout_result.bboxes:
        label = LABEL_MAP.get(box.label, "text")
        x0, y0, x1, y1 = box.bbox
        regions.append(Region(page=0, label=label, score=float(box.confidence), bbox=(float(x0), float(y0), float(x1), float(y1))))
    return regions


def _predict_layout_page(doc_stem: str, page_index: int, image_path: Path, client: SuryaClient):
    from PIL import Image

    image = Image.open(image_path)
    layout_result = client.predict_layout(image)
    return doc_stem, page_index, image, layout_result


def run_stage_layout(client: SuryaClient, input_dir: Path, output_dir: Path, visualize: bool, tmp_dir: Path, concurrency: int) -> None:
    docs = sorted(p for p in input_dir.iterdir() if p.suffix.lower() == ".pdf" or p.suffix.lower() in IMAGE_SUFFIXES)
    if not docs:
        print("[peyk-surya] no PDF/image files found in input", file=sys.stderr)
        return

    from tqdm import tqdm

    # Every page of every document is rendered up front (cheap, local pypdfium2 work, not the
    # bottleneck) before any network dispatch — predict_layout calls (the actual bottleneck)
    # then run concurrently across every page of every document at once, same batching
    # rationale as run_stage_ocr/tsr/table-full. Unlike those (one crop -> one independent
    # output file each), a document's pages must still be assembled back in page order
    # afterward — peyk-layout's own region order across the whole document is what final
    # assembly relies on (see pipeline.md) — so results are collected into a dict keyed by
    # (doc_stem, page_index) here and only assembled into pages_out, in order, once every
    # future for that document has completed, regardless of the order they actually finish in.
    page_tasks: list[tuple[str, int, Path]] = [
        (doc_path.stem, page_index, image_path)
        for doc_path in docs
        for page_index, image_path in iter_page_images(doc_path, tmp_dir)
    ]

    results: dict[tuple[str, int], tuple] = {}  # (doc_stem, page_index) -> (image, image_path, layout_result)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(_predict_layout_page, doc_stem, page_index, image_path, client): (doc_stem, page_index, image_path)
            for doc_stem, page_index, image_path in page_tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="[peyk-surya] layout", unit="page", file=sys.stderr):
            doc_stem, page_index, image_path = futures[future]
            try:
                _, _, image, layout_result = future.result()
                results[(doc_stem, page_index)] = (image, image_path, layout_result)
            except Exception as e:
                tqdm.write(f"[peyk-surya] {doc_stem} page {page_index} failed: {e}", file=sys.stderr)

    for doc_path in docs:
        doc_stem = doc_path.stem
        doc_page_indices = sorted(pi for (ds, pi) in results if ds == doc_stem)
        if not doc_page_indices:
            continue

        pages_out = []
        for page_index in doc_page_indices:
            image, image_path, layout_result = results[(doc_stem, page_index)]
            # Deliberately NOT applying peyk-layout/run.py's _reading_order() raster-sort
            # heuristic here — trusting whatever order Surya's own result returns, since that
            # claimed real reading order is the whole point of trying this backend (see
            # implementation_plan.md Task 1.8's intro).
            regions = _regions_from_layout_result(layout_result)
            for region in regions:
                region.page = page_index
            pages_out.extend(r.to_dict() for r in regions)

            # Persisted unconditionally, matching peyk-layout's own contract exactly —
            # peyk-orchestrator's pipeline.py load_rendered_pages() requires this file
            # regardless of which layout backend produced it.
            raw_path = output_dir / f"{doc_stem}_p{page_index}_raw.png"
            image.save(raw_path)

            if visualize:
                _draw_regions(image_path, regions, output_dir / f"{doc_stem}_p{page_index}.png")

        out_path = output_dir / f"{doc_stem}.json"
        out_path.write_text(json.dumps({"document": doc_path.name, "model": "surya", "regions": pages_out}, indent=2, ensure_ascii=False))
        print(f"[peyk-surya] wrote {out_path}", file=sys.stderr)


def _draw_regions(image_path: Path, regions: list[Region], out_path: Path) -> None:
    from PIL import Image, ImageDraw

    colors = {"text": (0, 128, 255), "table": (0, 200, 0), "figure": (220, 0, 0)}
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for region in regions:
        color = colors.get(region.label, (255, 165, 0))
        x0, y0, x1, y1 = region.bbox
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        draw.text((x0 + 2, max(0, y0 - 12)), f"{region.label} {region.score:.2f}", fill=color)
    image.save(out_path)


# ---------------------------------------------------------------------------
# --stage tsr
# ---------------------------------------------------------------------------


def _structure_from_table_rec_result(table_result) -> TableStructure:
    """Maps Surya's TableRecPredictor structure-only result onto this project's
    TableStructure/Cell dataclasses — see backends/client.py's caveat, same as layout."""
    cells = []
    for cell in table_result.cells:
        cells.append(
            Cell(
                row=cell.row_id,
                col=cell.col_id,
                row_span=getattr(cell, "rowspan", 1),
                col_span=getattr(cell, "colspan", 1),
                bbox=tuple(float(v) for v in cell.bbox),
            )
        )
    num_rows = max((c.row + c.row_span for c in cells), default=0)
    num_cols = max((c.col + c.col_span for c in cells), default=0)
    return TableStructure(num_rows=num_rows, num_cols=num_cols, cells=cells)


def _process_crop_tsr(crop_path: Path, client: SuryaClient, output_dir: Path, visualize: bool) -> None:
    from PIL import Image

    image = Image.open(crop_path)
    table_result = client.predict_table_structure(image)
    structure = _structure_from_table_rec_result(table_result)

    out_path = output_dir / f"{crop_path.stem}.json"
    out_path.write_text(json.dumps({"crop": crop_path.name, "model": "surya", **structure.to_dict()}, indent=2))

    rows = row_boxes(structure, image.width)
    cols = col_boxes(structure, image.width, image.height)
    cells = regularized_cells(structure, rows, cols)
    aug_path = output_dir / f"{crop_path.stem}_aug.json"
    aug_path.write_text(json.dumps({
        "crop": crop_path.name,
        "model": "surya",
        "rows": [r.to_dict() for r in rows],
        "cols": [c.to_dict() for c in cols],
        "cells": cells,
    }, indent=2))

    if visualize:
        _draw_cells(crop_path, structure, output_dir / f"{crop_path.stem}_viz.png")
        _draw_aug(crop_path, rows, cols, cells, output_dir / f"{crop_path.stem}_aug_viz.png")


def run_stage_tsr(client: SuryaClient, input_dir: Path, output_dir: Path, visualize: bool, concurrency: int) -> None:
    from tqdm import tqdm

    crops = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not crops:
        print("[peyk-surya] no crop images found in input", file=sys.stderr)
        return

    # Each crop's structure recognition is independent of every other's — same batching
    # rationale as run_stage_ocr (see its own docstring), now worth doing here too since
    # peyk-vllm-surya was confirmed to actually support real concurrent throughput (35x+ at
    # max-num-seqs=8) once GPU-memory tuning stopped forcing max-num-seqs down to 1-2 — see
    # implementation_plan.md Task 1.8.
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_process_crop_tsr, crop_path, client, output_dir, visualize): crop_path for crop_path in crops}
        for future in tqdm(as_completed(futures), total=len(futures), desc="[peyk-surya] TSR", unit="crop", file=sys.stderr):
            crop_path = futures[future]
            try:
                future.result()
            except Exception as e:
                tqdm.write(f"[peyk-surya] {crop_path.name} failed: {e}", file=sys.stderr)


def _draw_cells(image_path: Path, structure: TableStructure, out_path: Path) -> None:
    """Raw per-cell boxes straight from the model, matching peyk-tsr/run.py's draw_cells() —
    the unprocessed counterpart to _draw_aug()'s row/col/regularized-grid visualization."""
    from PIL import Image, ImageDraw

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for cell in structure.cells:
        x0, y0, x1, y1 = cell.bbox
        draw.rectangle([x0, y0, x1, y1], outline=(0, 200, 0), width=2)
        draw.text((x0 + 2, max(0, y0 - 12)), f"r{cell.row}c{cell.col}", fill=(220, 0, 0))
    image.save(out_path)


def _draw_aug(image_path: Path, rows, cols, cells: list[dict], out_path: Path) -> None:
    from PIL import Image, ImageDraw

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for row in rows:
        draw.rectangle(list(row.bbox), outline=(220, 130, 0), width=1)
    for col in cols:
        draw.rectangle(list(col.bbox), outline=(0, 100, 220), width=1)
    for cell in cells:
        draw.rectangle(cell["bbox"], outline=(0, 200, 0), width=2)
    image.save(out_path)


# ---------------------------------------------------------------------------
# --stage table-full
# ---------------------------------------------------------------------------
#
# Not the same code path as --stage tsr: this is peyk-orchestrator's dedicated dispatch for
# the tsr.backend=="surya" + cell_ocr resolving to "surya" case (see pipeline.py) — instead of TableRecPredictor's
# structure-only path + per-cell OCR (the every-other-backend-combination pattern --stage tsr
# matches), it calls predict_full: one call per table, structure AND text together, with real
# table-wide context instead of isolated per-cell crops. Output contract is deliberately
# different from --stage tsr's (`<crop-stem>.json` structure + `_aug.json`) — this writes
# ready-to-markdownify HTML directly, since peyk-orchestrator's table-routing logic for this
# case renders it straight into the assembled document rather than pairing structure with
# separately-sourced cell text.


def _html_from_table_full_result(result) -> str:
    """TableRecPredictor.predict_full's result shape is NOT confirmed the way
    RecognitionPredictor's PageOCRResult/BlockOCRResult shape now is (see
    _blocks_from_recognition_result's docstring for that investigation) — this guess
    (flat `.html`/`.text` attribute) hasn't been checked against Surya's real source or a live
    response. Expect to revisit this the same way _text_from_recognition_result's original
    guess had to be, once this stage actually runs against the live server."""
    return getattr(result, "html", None) or getattr(result, "text", "") or ""


def _process_crop_table_full(crop_path: Path, client: SuryaClient, output_dir: Path) -> None:
    from PIL import Image

    image = Image.open(crop_path)
    table_html_result = client.predict_table_full(image)
    html = _html_from_table_full_result(table_html_result)

    out_path = output_dir / f"{crop_path.stem}.json"
    out_path.write_text(json.dumps({"crop": crop_path.name, "model": "surya", "html": html}, indent=2, ensure_ascii=False))


def run_stage_table_full(client: SuryaClient, input_dir: Path, output_dir: Path, concurrency: int) -> None:
    from tqdm import tqdm

    crops = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not crops:
        print("[peyk-surya] no crop images found in input", file=sys.stderr)
        return

    # Each table's predict_full call is independent of every other table's — same batching
    # rationale as run_stage_ocr/run_stage_tsr. The 3-table celebrated end-to-end test
    # (implementation_plan.md Task 1.8) ran sequentially and took 21.78s doing so; this stands
    # to shrink meaningfully for documents with more tables once dispatched concurrently.
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_process_crop_table_full, crop_path, client, output_dir): crop_path for crop_path in crops}
        for future in tqdm(as_completed(futures), total=len(futures), desc="[peyk-surya] table-full", unit="table", file=sys.stderr):
            crop_path = futures[future]
            try:
                future.result()
            except Exception as e:
                tqdm.write(f"[peyk-surya] {crop_path.name} failed: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# --stage ocr
# ---------------------------------------------------------------------------


def _blocks_from_recognition_result(rec_result) -> list:
    """RecognitionPredictor.__call__ returns List[PageOCRResult] (confirmed against
    surya/recognition/__init__.py's real source — the earlier flat `.html`/`.confidence`
    guess on rec_result itself was wrong and silently produced empty text every time,
    since PageOCRResult has no such attributes). The real content lives one level down:
    each PageOCRResult has a `.blocks` list of BlockOCRResult, each with its own
    `.html`/`.confidence`/`.error`/`.skipped`. Since --stage ocr's predict_recognition()
    calls always pass a single image and no layout_result, they resolve through
    RecognitionPredictor's "full page" mode (full_page=True when layout_results is None) —
    our already-cropped region/cell image is treated as a tiny "page" and its content
    parsed into however many blocks the model's HTML output describes: 0 for a genuinely
    blank crop (a real, correct empty result, not a bug), 1 in the common case, occasionally
    more for a crop with visually distinct sub-regions. Shared here since fullpage mode's
    options 1/2/3/4 hit the exact same PageOCRResult/BlockOCRResult shape, whether or not
    layout_results was passed (block-mode vs. full-page mode both return this same type)."""
    return [
        block for block in (getattr(rec_result, "blocks", None) or [])
        if not getattr(block, "skipped", False) and not getattr(block, "error", False)
    ]


def _text_from_recognition_result(rec_result) -> OCRResult:
    """Maps Surya's RecognitionPredictor result onto OCRResult — plain-text (tags stripped),
    for --stage ocr's {"text", "score"} contract."""
    import re

    blocks = _blocks_from_recognition_result(rec_result)
    texts = [re.sub(r"<[^>]+>", "", getattr(b, "html", "") or "").strip() for b in blocks]
    confidences = [float(getattr(b, "confidence", 1.0)) for b in blocks]
    text = "\n".join(t for t in texts if t)
    score = sum(confidences) / len(confidences) if confidences else 1.0
    return OCRResult(text=text, score=score)


def _html_from_recognition_result(rec_result) -> str:
    """Same PageOCRResult/BlockOCRResult shape as _text_from_recognition_result, but joins
    raw HTML (not tag-stripped) for fullpage mode's markdownify() call sites, which want
    real HTML to convert, not plain text."""
    return "".join(getattr(b, "html", "") or "" for b in _blocks_from_recognition_result(rec_result))


# Shared default across every stage's concurrent dispatch (layout/tsr/ocr/table-full), not
# just OCR despite the name kept for now. Deliberately conservative next to peyk-paddleocr-vl's
# proven 32: this was the first time SuryaClient's underlying HTTP client was called
# concurrently at all, and unlike PaddleX's genai_client (documented thread-safe
# openai-SDK-style client), Surya's own thread-safety hasn't been independently confirmed here.
# Also deliberately matches peyk-vllm-surya's own --max-num-seqs=8 (see start.sh) so client and
# server concurrency stay in lockstep — raising one without the other either wastes client-side
# parallelism the server can't actually run concurrently, or oversubmits past what the server
# will admit at once. Raise both together via --concurrency/MAX_NUM_SEQS once real behavior (no
# corrupted/cross-talking responses under load) is confirmed at this level.
DEFAULT_OCR_CONCURRENCY = 8


def _process_crop_ocr(crop_path: Path, client: SuryaClient, output_dir: Path) -> None:
    from PIL import Image

    image = Image.open(crop_path)
    rec_result = client.predict_recognition(image)
    result = _text_from_recognition_result(rec_result)

    out_path = output_dir / f"{crop_path.stem}.json"
    out_path.write_text(json.dumps({"crop": crop_path.name, "model": "surya", **result.to_dict()}, indent=2, ensure_ascii=False))


def run_stage_ocr(client: SuryaClient, input_dir: Path, output_dir: Path, concurrency: int) -> None:
    from tqdm import tqdm

    crops = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not crops:
        print("[peyk-surya] no crop images found in input", file=sys.stderr)
        return

    # Crops dispatched concurrently, not one at a time — vLLM's whole performance model is
    # continuous batching, which only engages when multiple requests are in flight at once.
    # A prior version of this function looped sequentially (one blocking call per crop), so a
    # table with 100+ cells paid full per-request latency 100+ times over with none of vLLM's
    # batching ever exercised — same gap peyk-paddleocr-vl/run.py already fixed for that
    # container; see its own module docstring for the fuller rationale.
    #
    # A tqdm progress bar replaces the old per-crop "processing.../wrote..." print pair —
    # with several crops in flight across threads at once, per-crop lines interleaved into
    # unreadable noise; a single bar is the more useful signal here. tqdm.write() (not print)
    # for per-crop failures below, so an error doesn't corrupt the bar's in-place redraw.
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_process_crop_ocr, crop_path, client, output_dir): crop_path for crop_path in crops}
        for future in tqdm(as_completed(futures), total=len(futures), desc="[peyk-surya] OCR", unit="crop", file=sys.stderr):
            crop_path = futures[future]
            try:
                future.result()
            except Exception as e:
                tqdm.write(f"[peyk-surya] {crop_path.name} failed: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# --mode fullpage
# ---------------------------------------------------------------------------


def _predict_fullpage_page(doc_stem: str, page_index: int, image_path: Path, client: SuryaClient):
    from PIL import Image

    image = Image.open(image_path)
    rec_result = client.predict_recognition(image)
    return doc_stem, page_index, rec_result


def run_fullpage(client: SuryaClient, input_dir: Path, output_dir: Path, tmp_dir: Path, concurrency: int) -> None:
    """Fullpage mode, simplified down to the one shape worth keeping as a genuinely distinct,
    configurable behavior — docs/surya/high_level.md's original "option 1": zero layout
    inference, one RecognitionPredictor call per page, no structure at all. This is the
    cheapest possible mode and the only one that actually skips layout entirely, unlike the
    other three original options, which were all dropped after review concluded they didn't
    offer anything a user would pick over stage mode:
      - option 2 (layout + one block-mode recognition call, no TableRecPredictor at all) —
        tables never go through the dedicated table-structure prompt/model, so its table
        fidelity is presumed strictly worse than TableRecPredictor's; removed rather than kept
        as a config choice no one should pick.
      - option 3 (layout + TableRecPredictor structure + per-cell recognition) — sends
        isolated cell crops to RecognitionPredictor, the exact failure mode `cell_ocr`
        (peyk-orchestrator's config) exists to prevent; excluded on that basis alone.
      - option 4 (layout + TableRecPredictor.predict_full + recognition for non-table text) —
        functionally identical to peyk-orchestrator's stage-mode combination of
        tsr.backend=ocr.backend=surya (see pipeline.py's table-routing logic), just as one
        warm in-process run instead of three container dispatches — not a separate mode
        worth its own config surface once stage mode covers the same outcome.
    See implementation_plan.md Task 1.8 for the fuller reasoning behind dropping the other
    three.

    Pages dispatched concurrently across every document at once, same rationale and same
    order-preserving approach as run_stage_layout (see its own docstring) — arguably the
    highest-value case for this, since fullpage requests are the single largest ones this
    project sends (a whole page, not a crop) and a multi-page document previously paid full
    per-page latency sequentially with none of vLLM's batching engaged."""
    from tqdm import tqdm
    from markdownify import markdownify

    docs = sorted(p for p in input_dir.iterdir() if p.suffix.lower() == ".pdf" or p.suffix.lower() in IMAGE_SUFFIXES)
    if not docs:
        print("[peyk-surya] no PDF/image files found in input", file=sys.stderr)
        return

    page_tasks: list[tuple[str, int, Path]] = [
        (doc_path.stem, page_index, image_path)
        for doc_path in docs
        for page_index, image_path in iter_page_images(doc_path, tmp_dir)
    ]

    results: dict[tuple[str, int], object] = {}  # (doc_stem, page_index) -> rec_result
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(_predict_fullpage_page, doc_stem, page_index, image_path, client): (doc_stem, page_index)
            for doc_stem, page_index, image_path in page_tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="[peyk-surya] fullpage", unit="page", file=sys.stderr):
            doc_stem, page_index = futures[future]
            try:
                _, _, rec_result = future.result()
                results[(doc_stem, page_index)] = rec_result
            except Exception as e:
                tqdm.write(f"[peyk-surya] {doc_stem} page {page_index} failed: {e}", file=sys.stderr)

    for doc_path in docs:
        doc_stem = doc_path.stem
        doc_page_indices = sorted(pi for (ds, pi) in results if ds == doc_stem)
        if not doc_page_indices:
            continue

        page_fragments = []
        for page_index in doc_page_indices:
            html = _html_from_recognition_result(results[(doc_stem, page_index)])
            page_fragments.append(markdownify(html) if html else "")

        out_path = output_dir / f"{doc_stem}.md"
        out_path.write_text("\n\n".join(page_fragments), encoding="utf-8")
        print(f"[peyk-surya] wrote {out_path}", file=sys.stderr)


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Surya-OCR-2 (via a vLLM server) in stage or fullpage mode.")
    parser.add_argument("--model", default="surya", choices=["surya"], help="Backend to use (only one exists; kept for CLI-interface parity with other containers).")
    parser.add_argument(
        "--lang", default="arabic", choices=["arabic", "latin"],
        help="Script of the crops being processed (unused by this backend; accepted for CLI-interface parity with peyk-simple-ocr/peyk-paddleocr-vl, since peyk-orchestrator's dispatch_ocr_batch always passes --lang).",
    )
    parser.add_argument("--mode", default="stage", choices=["stage", "fullpage"], help="stage: matches one existing container's per-stage output contract. fullpage: zero layout inference, one recognition call per page (see run_fullpage's docstring for why the other three originally-planned fullpage shapes were dropped).")
    parser.add_argument("--stage", choices=["layout", "tsr", "ocr", "table-full"], help="Required when --mode stage. table-full: predict_full (structure+text in one call) for tables, dispatched by peyk-orchestrator only when tsr.backend==surya and cell_ocr resolves to surya too — see pipeline.py.")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL, help=f"Base URL of the peyk-vllm-surya server (default: {DEFAULT_SERVER_URL}).")
    parser.add_argument("--input", required=True, type=Path, help="Directory of input documents/images/crops (meaning depends on --mode/--stage).")
    parser.add_argument("--output", required=True, type=Path, help="Directory to write output.")
    parser.add_argument("--visualize", action="store_true", help="Also write a visualization PNG, for stage in {layout, tsr}.")
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_OCR_CONCURRENCY,
        help=f"Max in-flight requests to the vLLM server at once — applies to every --stage (layout/tsr/ocr/table-full) and to --mode fullpage (default: {DEFAULT_OCR_CONCURRENCY}).",
    )
    args = parser.parse_args()

    if args.mode == "stage" and args.stage is None:
        parser.error("--stage is required when --mode stage")

    if not args.input.is_dir():
        parser.error(f"--input {args.input} is not a directory")
    args.output.mkdir(parents=True, exist_ok=True)

    role_by_stage = {"layout": {"layout"}, "tsr": {"table_rec"}, "ocr": {"recognition"}, "table-full": {"table_rec"}}
    # Fullpage mode is down to one shape now (see run_fullpage's docstring) — always just
    # "recognition", no layout/table_rec roles to conditionally load anymore.
    roles = role_by_stage[args.stage] if args.mode == "stage" else {"recognition"}

    client = SuryaClient(server_url=args.server_url)
    print(f"[peyk-surya] connecting to {args.server_url} (roles: {sorted(roles)})...", file=sys.stderr)
    client.load(roles)
    print("[peyk-surya] client ready.", file=sys.stderr)

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        if args.mode == "stage":
            if args.stage == "layout":
                run_stage_layout(client, args.input, args.output, args.visualize, tmp_dir, args.concurrency)
            elif args.stage == "tsr":
                run_stage_tsr(client, args.input, args.output, args.visualize, args.concurrency)
            elif args.stage == "ocr":
                run_stage_ocr(client, args.input, args.output, args.concurrency)
            elif args.stage == "table-full":
                run_stage_table_full(client, args.input, args.output, args.concurrency)
        else:
            run_fullpage(client, args.input, args.output, tmp_dir, args.concurrency)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
