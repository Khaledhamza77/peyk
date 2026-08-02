#!/usr/bin/env python3
"""Direct Character Recognition CLI — born-digital text, no model.

Usage:
    run.py --input <dir> --output <dir>

--input must contain exactly one source PDF and a manifest.json, a list of two kinds of
entries:

  - Whole-region/whole-cell: {"id": "r0", "page": 0, "bbox": [x0, y0, x1, y1]} — extracts
    the PDF's own text within that single rectangle via get_text_bounded, same as before.

  - Table row (mode: "row"): {"mode": "row", "id": "r4_row2", "page": 0,
    "row_bbox": [x0, y0, x1, y1], "cols": [{"id": "r4_c8", "bbox": [...]}, ...]}. Instead of
    trusting any individual cell's bbox to bound a get_text_bounded call (which can clip
    content the table-structure model under-boxed), this walks every character in the row's
    y-band by its own real per-character position (pypdfium2's get_charbox) and buckets each
    one into whichever column's x-range contains it, writing one output per column id. Column
    bboxes are expected to already be contiguous/gap-free (peyk-tsr's col_boxes()) so every
    x-position belongs to exactly one bucket.

bbox is in the same raster-pixel space (top-left origin, RENDER_SCALE DPI) that
peyk-orchestrator's layout regions use — see pipeline.py's RENDER_SCALE.

Writes "<id>.json" per output (one per whole-region entry, or one per column per row entry),
matching the OCR containers' output shape ({"text": ...}) so peyk-orchestrator can assemble
either path the same way.
"""
import argparse
import json
import sys
from pathlib import Path

import pypdfium2 as pdfium

# Must match peyk-orchestrator's pipeline.py RENDER_SCALE — that's the DPI the layout
# bboxes were computed at (300/72), not the PDF's native 72-DPI point space.
RENDER_SCALE = 300 / 72


def _to_point_bounds(page: "pdfium.PdfPage", bbox: list[float]) -> tuple[float, float, float, float]:
    """Pixel-space (top-left origin, RENDER_SCALE DPI) -> PDF point-space (bottom-left
    origin), matching pypdfium2's own (left, bottom, right, top) convention."""
    x0, y0, x1, y1 = bbox
    left = x0 / RENDER_SCALE
    right = x1 / RENDER_SCALE
    _, page_height = page.get_size()
    top = page_height - (y0 / RENDER_SCALE)
    bottom = page_height - (y1 / RENDER_SCALE)
    return left, bottom, right, top


def region_text(page: "pdfium.PdfPage", bbox: list[float]) -> str:
    left, bottom, right, top = _to_point_bounds(page, bbox)
    textpage = page.get_textpage()
    return textpage.get_text_bounded(left=left, bottom=bottom, right=right, top=top)


def row_cell_texts(page: "pdfium.PdfPage", row_bbox: list[float], cols: list[dict]) -> dict[str, str]:
    row_left, row_bottom, row_right, row_top = _to_point_bounds(page, row_bbox)
    col_bounds = [(col["id"], _to_point_bounds(page, col["bbox"])) for col in cols]

    textpage = page.get_textpage()
    buckets: dict[str, list[str]] = {col_id: [] for col_id, _ in col_bounds}
    # Iterating in character-stream order (not re-sorted by x) preserves the PDF's own
    # logical reading order within each bucket — important for RTL (Arabic) text, where
    # stream order already comes out correctly formatted (confirmed empirically: plain
    # get_text_bounded already extracts Arabic correctly without any reordering on our
    # part), but sorting characters by x-position first would silently break that.
    for i in range(textpage.count_chars()):
        left, bottom, right, top = textpage.get_charbox(i)
        cx, cy = (left + right) / 2, (bottom + top) / 2
        if not (row_bottom <= cy <= row_top):
            continue
        for col_id, (c_left, c_bottom, c_right, c_top) in col_bounds:
            if c_left <= cx <= c_right:
                buckets[col_id].append(textpage.get_text_range(i, 1))
                break

    return {col_id: "".join(chars) for col_id, chars in buckets.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract born-digital text for a batch of layout regions.")
    parser.add_argument("--input", required=True, type=Path, help="Directory containing the source PDF and manifest.json.")
    parser.add_argument("--output", required=True, type=Path, help="Directory to write per-region extracted text.")
    args = parser.parse_args()

    if not args.input.is_dir():
        parser.error(f"--input {args.input} is not a directory")
    args.output.mkdir(parents=True, exist_ok=True)

    manifest_path = args.input / "manifest.json"
    if not manifest_path.is_file():
        parser.error(f"no manifest.json found in {args.input}")
    manifest = json.loads(manifest_path.read_text())

    pdf_candidates = [p for p in args.input.iterdir() if p.suffix.lower() == ".pdf"]
    if len(pdf_candidates) != 1:
        parser.error(f"expected exactly one PDF in {args.input}, found {len(pdf_candidates)}")
    pdf_path = pdf_candidates[0]

    pdf = pdfium.PdfDocument(str(pdf_path))
    pages: dict[int, "pdfium.PdfPage"] = {}
    try:
        for entry in manifest:
            page_num = entry["page"]
            if page_num not in pages:
                pages[page_num] = pdf[page_num]
            page = pages[page_num]

            if entry.get("mode") == "row":
                print(f"[peyk-dcr] extracting row {entry['id']} (page {page_num}, {len(entry['cols'])} cols)...", file=sys.stderr)
                texts = row_cell_texts(page, entry["row_bbox"], entry["cols"])
                for col in entry["cols"]:
                    out_path = args.output / f"{col['id']}.json"
                    out_path.write_text(json.dumps({"text": texts.get(col["id"], "")}, indent=2, ensure_ascii=False))
                    print(f"[peyk-dcr] wrote {out_path}", file=sys.stderr)
            else:
                entry_id = entry["id"]
                print(f"[peyk-dcr] extracting {entry_id} (page {page_num})...", file=sys.stderr)
                text = region_text(page, entry["bbox"])
                out_path = args.output / f"{entry_id}.json"
                out_path.write_text(json.dumps({"text": text}, indent=2, ensure_ascii=False))
                print(f"[peyk-dcr] wrote {out_path}", file=sys.stderr)
    finally:
        pdf.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
