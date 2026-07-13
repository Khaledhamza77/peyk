"""Core dispatch: peyk-layout -> born-digital check -> per-region stage dispatch ->
assembly. Regions are handed to their stage in `peyk-layout`'s own output order and
reassembled in that same order (see pipeline.md — no separate reading-order solver)."""
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

from config import PipelineConfig, StageConfig, full_table_backend, is_vlm_model, vlm_provider
from stages import run_docker_stage, stub_fragment

RENDER_SCALE = 300 / 72  # must match peyk-layout's render scale so region bboxes line up

# OCR crops used to be re-cropped from a second, higher-DPI (600) render on the theory that
# paddleocr-vl needed more resolution on small/dense cell content. A real scale-sweep test
# against the live server (0.5x-3x) mostly falsified that (implementation_plan.md Task 1.5) —
# and later, further use found paddleocr-vl actually does *worse* at the higher DPI on some
# content, not just "no better." Removed entirely: every crop (whole-region or table-cell OCR
# included) is now cropped straight from the same RENDER_SCALE render everything else uses —
# one less render pass per document, and no separate high-DPI path to feed a backend that
# doesn't want it.

# Eastern Arabic-Indic (U+0660-0669), Persian/Extended Arabic-Indic (U+06F0-06F9), and ASCII
# digits can all show up in the same document — peyk-dcr passes through whatever glyphs the
# PDF's own text layer uses, and different OCR/VLM models pick different scripts even on the
# same source (real example seen in this project: a full-table VLM response rendering one
# number as "171,464,7١٢", ASCII and Eastern Arabic-Indic digits within the same number).
# normalize_digits() only touches a number when it's actually MIXED across scripts — a number
# already consistent in one script (whichever it is) is left as-is, since that's not an error
# to fix, just the source's own choice. Only a mixed number gets normalized, to `lang`'s target
# script: Eastern Arabic-Indic for "arabic", ASCII for anything else (matches StageConfig.lang).
_ASCII_DIGITS = "0123456789"
_EASTERN_ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
_ALL_DIGITS = _ASCII_DIGITS + _EASTERN_ARABIC_INDIC_DIGITS + _PERSIAN_DIGITS
_DIGIT_SCRIPT = {ch: "ascii" for ch in _ASCII_DIGITS}
_DIGIT_SCRIPT.update({ch: "arabic" for ch in _EASTERN_ARABIC_INDIC_DIGITS})
_DIGIT_SCRIPT.update({ch: "persian" for ch in _PERSIAN_DIGITS})
# A "number" for mixed-script detection purposes: digits plus the separators ("," ".") commonly
# seen between digit groups in this project's real output (thousands separators, decimals) —
# not full Unicode-number-formatting support, just enough to span one financial figure.
_NUMBER_RUN_RE = re.compile(rf"[{_ALL_DIGITS}](?:[{_ALL_DIGITS},.]*[{_ALL_DIGITS}])?")


def normalize_digits(text: str, lang: str = "arabic") -> str:
    target_digits = _EASTERN_ARABIC_INDIC_DIGITS if lang == "arabic" else _ASCII_DIGITS
    translation = str.maketrans(_ALL_DIGITS, target_digits * 3)

    def _normalize_if_mixed(match: re.Match) -> str:
        run = match.group(0)
        scripts = {_DIGIT_SCRIPT[ch] for ch in run if ch in _DIGIT_SCRIPT}
        return run.translate(translation) if len(scripts) > 1 else run

    return _NUMBER_RUN_RE.sub(_normalize_if_mixed, text)


def load_rendered_pages(doc_stem: str, layout_out_dir: Path) -> dict[int, Path]:
    """Loads peyk-layout's own raw per-page renders (see that container's run.py) instead of
    rasterizing the PDF a second time at the same RENDER_SCALE DPI for cropping — peyk-layout
    already renders every page at this exact scale for its own inference; re-rendering here
    was pure duplicate work repeated on every document (see docs/build_notes.md's efficiency
    review). This is the only render pass now — every crop (OCR included) comes from it."""
    prefix = f"{doc_stem}_p"
    pages = {}
    for path in layout_out_dir.glob(f"{prefix}*_raw.png"):
        page_idx = int(path.stem.removeprefix(prefix).removesuffix("_raw"))
        pages[page_idx] = path
    return pages


def render_pdf_pages(doc_path: Path, out_dir: Path) -> dict[int, Path]:
    """Rasterizes doc_path's pages directly via pypdfium2 at RENDER_SCALE, same call
    peyk-layout's own iter_page_images() makes (`page.render(scale=RENDER_SCALE).to_pil()`) —
    used only by the fullpage job (config.fullpage), which has no region-detection use for
    peyk-layout's actual model inference at all. Running the full peyk-layout container
    (a real GPU model pass over every page) purely to get its raw-render side effect would be
    paying for and discarding the one part of that container's output this job doesn't need —
    rendering itself is plain rasterization, no model involved, so it's done here directly
    instead. Every other job still gets its renders from peyk-layout's own output
    (load_rendered_pages above) since they need its region detection anyway."""
    pages = {}
    pdf = pdfium.PdfDocument(str(doc_path))
    try:
        for page_index in range(len(pdf)):
            image = pdf[page_index].render(scale=RENDER_SCALE).to_pil()
            out_path = out_dir / f"{doc_path.stem}_p{page_index}_raw.png"
            image.save(out_path)
            pages[page_index] = out_path
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


def crop_region(page_image: Image.Image, bbox: list[float], out_path: Path, padding_px: int = CROP_PADDING_PX) -> tuple[int, int]:
    """Returns the (x0, y0) page-pixel origin of the saved crop — needed later to translate
    peyk-tsr's cell bboxes (local to the crop) back into page-pixel space.

    Takes an already-opened Image, not a path — callers are expected to open (and cache) each
    page once via _cached_page_image() and reuse it across every crop taken from that page.
    A table with 100+ cells routed through the per-cell OCR crop path previously called
    Image.open() on the same multi-megapixel page PNG once per cell, which was real, avoidable
    decode overhead repeated for every single cell instead of once per page."""
    x0, y0, x1, y1 = bbox
    x0 = max(0, int(x0) - padding_px)
    y0 = max(0, int(y0) - padding_px)
    x1 = min(page_image.width, int(x1) + padding_px)
    y1 = min(page_image.height, int(y1) + padding_px)
    page_image.crop((x0, y0, x1, y1)).save(out_path)
    return x0, y0


def _cached_page_image(path: Path, cache: dict[Path, Image.Image]) -> Image.Image:
    image = cache.get(path)
    if image is None:
        image = Image.open(path)
        image.load()  # Image.open() is lazy; force the decode now so it happens exactly once
        cache[path] = image
    return image


def run_layout(config: PipelineConfig, input_dir: Path, workdir: Path) -> dict[str, dict]:
    """Batch layout over every document in input_dir. Returns {doc_stem: regions_json}."""
    out_dir = workdir / "layout_out"
    # peyk-surya's run.py needs an explicit --stage layout to know which of its three roles
    # to serve — every other layout model's image only ever does one thing, so this is a
    # no-op for them (see implementation_plan.md Task 1.8).
    extra_args = ["--visualize"]
    if config.layout.backend == "surya":
        extra_args += ["--stage", "layout"]
    run_docker_stage(
        image=config.layout.image,
        model=config.layout.backend,
        input_dir=input_dir,
        output_dir=out_dir,
        extra_args=extra_args,
    )
    return {p.stem: json.loads(p.read_text()) for p in out_dir.glob("*.json")}


def assemble_markdown_table(structure: dict, cell_texts: dict[int, str], lang: str = "arabic") -> str:
    """Render a peyk-tsr cell grid + paired-in text as a markdown table. Simplification:
    every cell is placed at its (row, col) top-left position only — a spanning cell's text
    doesn't get repeated into the extra rows/cols it covers, since markdown's own table syntax
    has no spanning-cell concept to render into anyway. Row 0 is emitted as the markdown
    header row purely because markdown tables require one; it's not necessarily a semantic
    header in the source table."""
    num_rows, num_cols = structure["num_rows"], structure["num_cols"]
    if num_rows == 0 or num_cols == 0:
        return ""

    grid = [["" for _ in range(num_cols)] for _ in range(num_rows)]
    for cell in structure["cells"]:
        text = normalize_digits(cell_texts.get(cell["cell_i"], ""), lang).replace("|", "\\|").replace("\n", " ").strip()
        grid[cell["row"]][cell["col"]] = text

    lines = ["| " + " | ".join(grid[0]) + " |", "| " + " | ".join(["---"] * num_cols) + " |"]
    for row in grid[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _vlm_credential_docker_args(backend: str) -> list[str]:
    """peyk-vlm's cloud credentials (containers/peyk-vlm/.env for Bedrock,
    containers/peyk-vlm/gcp-key.json for Vertex) are fixed, local-dev-only files living in the
    repo — not runtime workdir paths inside the orchestrator's own /hotstorage mounts, so
    --volumes-from (stages.py) doesn't help here. run_local.sh resolves both to host-absolute
    paths and passes them in as PEYK_VLM_ENV_FILE/PEYK_VLM_GCP_KEY_FILE env vars; these are
    then used as literal host paths in the *inner* `docker run` below, which reaches the host
    daemon directly over the shared socket (same reasoning stages.py's run_docker_stage
    docstring already documents for input_dir/output_dir, just via a fixed config-known path
    here instead of a dynamic one). Raises loudly if the one actually needed isn't set, rather
    than silently dispatching peyk-vlm without credentials. Used by any job that routes to a
    peyk-vlm model — ocr, tsr (full-table combination), figures, fullpage alike.

    Which credential file to use is decided by the model's real provider (config.vlm_provider,
    queried from peyk-vlm's own registry — see config.py's _vlm_models), not by guessing from
    the model key's spelling: "bedrock" needs AWS credentials, "vertex-gemini"/"vertex-maas"
    both need the same GCP credentials (they're the same cloud, just different Vertex APIs)."""
    provider = vlm_provider(backend)
    if provider == "bedrock":
        env_file = os.environ.get("PEYK_VLM_ENV_FILE")
        if not env_file:
            raise ValueError(
                f"model {backend!r} is a Bedrock model but PEYK_VLM_ENV_FILE isn't "
                "set — see containers/peyk-vlm/README.md (generate a Bedrock API key) and "
                "run_local.sh (which resolves and passes this env var)."
            )
        return ["--env-file", env_file]
    if provider in ("vertex-gemini", "vertex-maas"):
        key_file = os.environ.get("PEYK_VLM_GCP_KEY_FILE")
        if not key_file:
            raise ValueError(
                f"model {backend!r} is a Vertex model but PEYK_VLM_GCP_KEY_FILE "
                "isn't set — see containers/peyk-vlm/README.md (create a service-account key) "
                "and run_local.sh (which resolves and passes this env var)."
            )
        return ["-v", f"{key_file}:/secrets/gcp-key.json:ro", "-e", "GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcp-key.json"]
    raise ValueError(f"Unknown peyk-vlm provider {provider!r} for model {backend!r}")


def prepare_document(
    doc_path: Path,
    regions_doc: dict,
    config: PipelineConfig,
    workdir: Path,
    ocr_in: Path,
    tsr_batch: list[tuple[str, str, Path]],
    table_full_batch: list[tuple[str, str, Path]],
    ocr_batch: list[tuple[str, str]],
    figures_in: Path,
    figures_batch: list[tuple[str, str]],
) -> dict:
    """Phase 1 (per document): render pages, crop every region, and collect this document's
    table/OCR/figure crops into the batch-wide tsr_batch/table_full_batch/ocr_batch/
    figures_batch pools (mutated in place) instead of dispatching them here. tsr_batch/
    table_full_batch entries are (doc_stem, local_id, crop_path) — each batch's own dispatch
    dir is populated later (dispatch_tsr_batch/dispatch_table_full_batch), so crops still live
    under this document's own crops_dir until then. Every table region goes to exactly one of
    the two batches, never both — full_table_backend(config) decides which, per document-wide
    config, not per-region (see dispatch_documents/config.py's full_table_backend). ocr_batch/
    figures_batch entries are just (doc_stem, local_id): crops are written directly into the
    shared ocr_in/figures_in with their final batch-ready name below, so there's nothing left
    to copy later (see dispatch_ocr_batch's docstring for why that copy used to exist and why
    it was real, avoidable waste for a table with 100+ cells)."""
    doc_stem = doc_path.stem
    # Reuses peyk-layout's own raw per-page renders (workdir/layout_out — see that container's
    # run.py) rather than rasterizing the same PDF pages a second time at the same RENDER_SCALE
    # DPI purely for cropping. Also the only render pass now — OCR crops come from it too (see
    # RENDER_SCALE's comment above for why the old second, higher-DPI OCR render was dropped).
    pages = load_rendered_pages(doc_stem, workdir / "layout_out")

    born_digital = born_digital_pages(doc_path, config.born_digital_min_chars, config.force_scanned)

    regions = regions_doc["regions"]
    crops_dir = workdir / "crops" / doc_stem
    # Cleared, not just mkdir(exist_ok=True): a stale crop left over from a previous run at
    # this same path would otherwise get fed into a stage as if it belonged to this run.
    if crops_dir.exists():
        shutil.rmtree(crops_dir)
    crops_dir.mkdir(parents=True, exist_ok=True)

    # One decode per distinct page path, reused across every crop taken from it (a table's
    # worth of per-cell OCR crops all come from the same page image) — see crop_region()'s
    # docstring for why this matters.
    page_image_cache: dict[Path, Image.Image] = {}

    route_tables_to_full = full_table_backend(config) is not None

    dcr_targets = []  # peyk-dcr manifest entries (whole-region, or row-mode for table rows)
    tsr_indices = []  # region_idx — table regions needing structure recognition
    table_full_indices = []  # region_idx — table regions routed to the full-table path instead
    for idx, region in enumerate(regions):
        region["_born_digital"] = born_digital.get(region["page"], False)
        # Born-digital text is extracted straight from the PDF's own text layer (no image
        # needed at all) — every other label gets cropped.
        if region["label"] == "table":
            page_image = _cached_page_image(pages[region["page"]], page_image_cache)
            crop_path = crops_dir / f"r{idx}.png"
            region["_crop_origin"] = crop_region(page_image, region["bbox"], crop_path)
            if route_tables_to_full:
                table_full_indices.append(idx)
                table_full_batch.append((doc_stem, f"r{idx}", crop_path))
            else:
                tsr_indices.append(idx)
                tsr_batch.append((doc_stem, f"r{idx}", crop_path))
        elif region["label"] == "text":
            if region["_born_digital"]:
                dcr_targets.append({"id": f"r{idx}", "page": region["page"], "bbox": region["bbox"]})
            else:
                page_image = _cached_page_image(pages[region["page"]], page_image_cache)
                local_id = f"r{idx}"
                crop_region(page_image, region["bbox"], ocr_in / f"{doc_stem}__{local_id}.png")
                ocr_batch.append((doc_stem, local_id))
        elif region["label"] == "figure":
            page_image = _cached_page_image(pages[region["page"]], page_image_cache)
            local_id = f"r{idx}"
            crop_region(page_image, region["bbox"], figures_in / f"{doc_stem}__{local_id}.png")
            figures_batch.append((doc_stem, local_id))

    return {
        "doc_stem": doc_stem,
        "regions": regions,
        "pages": pages,
        "page_image_cache": page_image_cache,
        "dcr_targets": dcr_targets,
        "tsr_indices": tsr_indices,
        "table_full_indices": table_full_indices,
    }


def dispatch_tsr_batch(tsr_batch: list[tuple[str, str, Path]], config: PipelineConfig, workdir: Path) -> Path | None:
    """Phase 2 (once, across every document): one `docker run` for every table crop
    collected in prepare_document, instead of one per document. peyk-tsr already accepts an
    arbitrary batch of crop images in one invocation (see its run.py) — dispatching it once
    per document was paying its model-load cold start N times for an N-document batch with
    no benefit, the same waste run_layout already avoids by running peyk-layout once for the
    whole input directory. Crops are named "<doc_stem>__<local_id>" in the shared input dir
    so filenames stay unique across documents; process_tsr_results reads them back the same
    way."""
    if not tsr_batch:
        return None
    tsr_in = workdir / "tsr_in"
    if tsr_in.exists():
        shutil.rmtree(tsr_in)
    tsr_in.mkdir(parents=True, exist_ok=True)
    for doc_stem, local_id, crop_path in tsr_batch:
        dest = tsr_in / f"{doc_stem}__{local_id}{crop_path.suffix}"
        dest.write_bytes(crop_path.read_bytes())
    tsr_out = workdir / "tsr_out"
    extra_args = ["--visualize"]
    if config.tsr.backend == "surya":
        extra_args += ["--stage", "tsr"]
    run_docker_stage(
        image=config.tsr.image,
        model=config.tsr.backend,
        input_dir=tsr_in,
        output_dir=tsr_out,
        extra_args=extra_args,
    )
    return tsr_out


def dispatch_table_full_batch(
    table_full_batch: list[tuple[str, str, Path]], config: PipelineConfig, workdir: Path
) -> dict[str, dict[str, str]]:
    """The tsr/cell_ocr-both-resolve-to-the-same-full-table-capable-model routing case
    (config.py's full_table_backend): one call per table (structure+text together, real
    table-wide context) — peyk-surya's --stage table-full, or peyk-vlm's --role table —
    instead of the usual structure-then-per-cell-OCR split dispatch_tsr_batch/
    process_tsr_results handle for every other model combination. Note ocr.model (whole-region
    text) plays no part in this condition — non-table text always goes through
    dispatch_ocr_batch/config.ocr regardless of whether tables were routed here. Returns
    {doc_stem: {local_id: html}} — ready HTML per table, rendered directly by
    assemble_document rather than paired with separately-sourced cell text. config.tsr.image/
    backend are reused here (guaranteed to match full_table_backend's own resolved model)
    rather than adding a third stage config for this."""
    if not table_full_batch:
        return {}
    table_full_in = workdir / "table_full_in"
    if table_full_in.exists():
        shutil.rmtree(table_full_in)
    table_full_in.mkdir(parents=True, exist_ok=True)
    for doc_stem, local_id, crop_path in table_full_batch:
        dest = table_full_in / f"{doc_stem}__{local_id}{crop_path.suffix}"
        dest.write_bytes(crop_path.read_bytes())
    table_full_out = workdir / "table_full_out"
    backend = config.tsr.backend
    if backend == "surya":
        extra_args, extra_docker_args = ["--stage", "table-full"], None
    else:
        extra_args, extra_docker_args = ["--role", "table"], _vlm_credential_docker_args(backend)
    run_docker_stage(
        image=config.tsr.image,
        model=backend,
        input_dir=table_full_in,
        output_dir=table_full_out,
        extra_args=extra_args,
        extra_docker_args=extra_docker_args,
        gpu=(backend == "surya"),
    )
    results: dict[str, dict[str, str]] = {}
    for json_path in table_full_out.glob("*.json"):
        doc_stem, local_id = json_path.stem.split("__", 1)
        results.setdefault(doc_stem, {})[local_id] = json.loads(json_path.read_text())["html" if backend == "surya" else "text"]
    return results


def process_tsr_results(
    doc_state: dict,
    tsr_out: Path | None,
    config: PipelineConfig,
    cell_ocr_config: StageConfig,
    cell_ocr_in: Path,
    cell_ocr_batch: list[tuple[str, str]],
) -> dict[int, dict]:
    """Phase 3 (per document): fold this document's slice of the batched TSR output back into
    born-digital row-mode DCR targets (appended to doc_state's own dcr_targets, dispatched
    per-document in phase 5) or scanned per-cell OCR targets (cropped into cell_ocr_in and
    appended to cell_ocr_batch). cell_ocr_config/cell_ocr_in/cell_ocr_batch are the same
    ocr_in/ocr_batch whole-region crops use when no cell_ocr override is configured (cells and
    whole-region text share one dispatch, as before) — see dispatch_documents for which case
    applies. When a cell_ocr override IS configured, these are a separate dir/batch/dispatch,
    keeping isolated cell crops off whatever (possibly VLM-style) model `ocr.model` uses."""
    tsr_structures: dict[int, dict] = {}
    if tsr_out is None or not doc_state["tsr_indices"]:
        return tsr_structures

    doc_stem = doc_state["doc_stem"]
    regions = doc_state["regions"]
    dcr_targets = doc_state["dcr_targets"]

    # Timing instrumentation: this whole function is pure Python running between the TSR
    # container returning and the OCR container starting — no docker/network calls of its own
    # — so if the pipeline feels slow specifically in that gap, this is where to look. Broken
    # into per-table JSON-parse/coordinate-translate time vs. crop+save time, since those have
    # very different profiles (cheap dict work vs. PIL image I/O per cell).
    t_func_start = time.perf_counter()
    total_json_s = 0.0
    total_crop_s = 0.0
    total_cells = 0

    for idx in doc_state["tsr_indices"]:
        struct_path = tsr_out / f"{doc_stem}__r{idx}.json"
        # "_aug.json" is peyk-tsr's row/column/regularized-cell calibration (base.py's
        # row_boxes/col_boxes/regularized_cells) — used here instead of the model's raw
        # per-cell bbox for both cropping (OCR) and row extraction (DCR), since the raw
        # per-cell box can clip or miss real content (see implementation_plan.md Task 1.5's
        # row/column-based OCR+DCR follow-up).
        aug_path = tsr_out / f"{doc_stem}__r{idx}_aug.json"
        if not struct_path.exists() or not aug_path.exists():
            continue
        t0 = time.perf_counter()
        data = json.loads(struct_path.read_text())
        aug = json.loads(aug_path.read_text())
        origin_x, origin_y = regions[idx]["_crop_origin"]
        region = regions[idx]

        def to_page(bbox: list[float], origin_x: float = origin_x, origin_y: float = origin_y) -> list[float]:
            x0, y0, x1, y1 = bbox
            return [origin_x + x0, origin_y + y0, origin_x + x1, origin_y + y1]

        row_page_bbox = {r["row"]: to_page(r["bbox"]) for r in aug["rows"]}

        cells = []
        cells_by_row: dict[int, list[dict]] = {}
        for cell_i, (raw_cell, reg_cell) in enumerate(zip(data["cells"], aug["cells"])):
            cell_id = f"r{idx}_c{cell_i}"
            cells.append({"row": raw_cell["row"], "col": raw_cell["col"], "cell_i": cell_i})
            cells_by_row.setdefault(raw_cell["row"], []).append(
                {"id": cell_id, "bbox": to_page(reg_cell["bbox"])}
            )
        tsr_structures[idx] = {"num_rows": data["num_rows"], "num_cols": data["num_cols"], "cells": cells}
        total_json_s += time.perf_counter() - t0

        if region["_born_digital"]:
            # One row-mode manifest entry per row: peyk-dcr buckets real per-character
            # positions (not any cell's own bbox) into each column, so a cell whose
            # structure-model box came out too narrow doesn't lose content at its edges.
            for row_idx, row_cells in cells_by_row.items():
                row_bbox = row_page_bbox.get(row_idx)
                if row_bbox is None:
                    continue
                dcr_targets.append({
                    "mode": "row",
                    "id": f"r{idx}_row{row_idx}",
                    "page": region["page"],
                    "row_bbox": row_bbox,
                    "cols": row_cells,
                })
        else:
            # Regularized (row-band x column-band) bbox used for the crop itself, instead
            # of the model's raw per-cell box — fixes overcropped/undercropped cells without
            # needing to split OCR text back into columns after the fact. Cropped from the
            # same RENDER_SCALE page render every other stage uses (see RENDER_SCALE's
            # comment above for why the old higher-DPI OCR-only render was dropped).
            t_crop0 = time.perf_counter()
            page_image = _cached_page_image(doc_state["pages"][region["page"]], doc_state["page_image_cache"])
            for row_cells in cells_by_row.values():
                for cell_entry in row_cells:
                    total_cells += 1
                    cell_crop_path = cell_ocr_in / f"{doc_stem}__{cell_entry['id']}.png"
                    crop_region(page_image, cell_entry["bbox"], cell_crop_path)
                    cell_ocr_batch.append((doc_stem, cell_entry["id"]))
            total_crop_s += time.perf_counter() - t_crop0

    print(
        f"[peyk-orchestrator] process_tsr_results({doc_stem}): {time.perf_counter() - t_func_start:.2f}s total "
        f"(json/translate: {total_json_s:.2f}s, crop+save: {total_crop_s:.2f}s over {total_cells} cells)",
        file=sys.stderr,
    )

    return tsr_structures


def dispatch_ocr_batch(
    ocr_batch: list[tuple[str, str]], stage_config: StageConfig, workdir: Path, ocr_in: Path, out_dir_name: str = "ocr_out"
) -> dict[str, dict[str, str]]:
    """Phase 4 (once, across every document): one `docker run` for every crop collected so
    far, instead of one per document — same batching rationale as dispatch_tsr_batch. Returns
    {doc_stem: {local_id: text}}.

    Takes stage_config explicitly (not the whole PipelineConfig) so this same function serves
    both the whole-region OCR dispatch (config.ocr) and, when a cell_ocr override is
    configured, a second, separate dispatch for table-cell crops (config.cell_ocr) — see
    dispatch_documents. out_dir_name distinguishes the two when both run (ocr_out vs.
    cell_ocr_out), so a batch run doesn't overwrite the other's output.

    Unlike dispatch_tsr_batch, there's no copy step here: ocr_in was already populated with
    every crop, already named "<doc_stem>__<local_id>.png", directly by prepare_document/
    process_tsr_results — previously this function built ocr_in itself by re-copying
    (read_bytes/write_bytes) each crop from its own per-document file, a second disk write for
    every single crop that mattered once a table had 100+ cells."""
    if not ocr_batch:
        return {}
    backend = stage_config.backend
    is_vlm = is_vlm_model(backend)
    if is_vlm:
        # peyk-vlm's CLI has no --lang/--server-url (those are peyk-simple-ocr/peyk-surya
        # concepts) — --role ocr is its equivalent of the Surya --stage ocr special-case below,
        # and it needs its own credential args instead of a GPU flag (no local GPU at all).
        ocr_extra_args = ["--role", "ocr"]
        extra_docker_args = _vlm_credential_docker_args(backend)
    else:
        ocr_extra_args = ["--lang", stage_config.lang]
        if stage_config.server_url:
            ocr_extra_args += ["--server-url", stage_config.server_url]
        if backend == "surya":
            ocr_extra_args += ["--stage", "ocr"]
        extra_docker_args = None
    ocr_out = workdir / out_dir_name
    run_docker_stage(
        image=stage_config.image,
        model=backend,
        input_dir=ocr_in,
        output_dir=ocr_out,
        extra_args=ocr_extra_args,
        extra_docker_args=extra_docker_args,
        gpu=not is_vlm,
    )
    results: dict[str, dict[str, str]] = {}
    for json_path in ocr_out.glob("*.json"):
        doc_stem, local_id = json_path.stem.split("__", 1)
        results.setdefault(doc_stem, {})[local_id] = json.loads(json_path.read_text())["text"]
    return results


def dispatch_figures_batch(figures_batch: list[tuple[str, str]], config: PipelineConfig, workdir: Path, figures_in: Path) -> dict[str, dict[str, str]]:
    """The figures job (implementation_plan.md Task 1.7's peyk-vlm wiring; renamed from the old
    standalone vlm: section) — structurally identical to dispatch_ocr_batch (one docker run for
    the whole batch, crops already written directly into figures_in by prepare_document with
    their final "<doc_stem>__<local_id>.png" name, no gpu). Returns
    {doc_stem: {local_id: description}}."""
    if not figures_batch:
        return {}
    figures_out = workdir / "figures_out"
    run_docker_stage(
        image=config.figures.image,
        model=config.figures.backend,
        input_dir=figures_in,
        output_dir=figures_out,
        extra_args=["--role", "figure"],
        extra_docker_args=_vlm_credential_docker_args(config.figures.backend),
        gpu=False,
    )
    results: dict[str, dict[str, str]] = {}
    for json_path in figures_out.glob("*.json"):
        doc_stem, local_id = json_path.stem.split("__", 1)
        results.setdefault(doc_stem, {})[local_id] = json.loads(json_path.read_text())["text"]
    return results


def dispatch_dcr(doc_path: Path, dcr_targets: list[dict], config: PipelineConfig, workdir: Path) -> dict[str, str]:
    """Phase 5 (per document — deliberately not batched, unlike TSR/OCR above): peyk-dcr
    loads no model (pure pypdfium2 text extraction), so its per-container startup cost is
    negligible regardless of how many times it's invoked. Its CLI is also scoped to exactly
    one source PDF per invocation (see containers/peyk-dcr/run.py) — batching it across
    documents would mean reworking that manifest contract to carry a per-entry document
    reference, for a cold-start saving that doesn't exist here."""
    if not dcr_targets:
        return {}
    doc_stem = doc_path.stem
    dcr_in = workdir / "dcr_in" / doc_stem
    if dcr_in.exists():
        shutil.rmtree(dcr_in)
    dcr_in.mkdir(parents=True, exist_ok=True)
    (dcr_in / doc_path.name).write_bytes(doc_path.read_bytes())
    (dcr_in / "manifest.json").write_text(json.dumps(dcr_targets))
    dcr_out = workdir / "dcr_out" / doc_stem
    run_docker_stage(
        image=config.dcr.image,
        model=None,
        input_dir=dcr_in,
        output_dir=dcr_out,
        gpu=False,
    )
    return {p.stem: json.loads(p.read_text())["text"] for p in dcr_out.glob("*.json")}


def assemble_document(
    doc_state: dict,
    tsr_structures: dict[int, dict],
    dcr_results: dict[str, str],
    ocr_results: dict[str, str],
    table_full_results: dict[str, str],
    figures_results: dict[str, str],
    lang: str = "arabic",
) -> str:
    """Phase 6 (per document): concatenate each region's fragment in peyk-layout's own
    region order — see pipeline.md, no separate reading-order solver. `lang` drives
    normalize_digits()'s target script for any number found mixing digit scripts — see that
    function's docstring; matches config.ocr.lang, this project's single per-run language
    setting (no per-region language detection exists)."""
    from markdownify import markdownify

    fragments = []
    for idx, region in enumerate(doc_state["regions"]):
        label = region["label"]
        if label == "text":
            region_id = f"r{idx}"
            if region["_born_digital"]:
                text = dcr_results.get(region_id)
                fragments.append(normalize_digits(text, lang) if text is not None else stub_fragment("peyk-dcr", "text"))
            else:
                text = ocr_results.get(region_id)
                fragments.append(normalize_digits(text, lang) if text is not None else stub_fragment("peyk-ocr", "text"))
        elif label == "table":
            region_id = f"r{idx}"
            full_html = table_full_results.get(region_id)
            if full_html is not None:
                # tsr/cell_ocr both resolving to the same full-table-capable model (config.py's
                # full_table_backend) — dispatch_table_full_batch already produced ready HTML
                # for this table in one call; tsr_structures/cell pairing below never ran for
                # this region at all (see prepare_document's route_tables_to_full branch).
                # normalize_digits(): the classical per-cell path (assemble_markdown_table)
                # already normalizes each cell's text; this full-table HTML path bypassed that
                # entirely until now, letting Eastern Arabic-Indic/ASCII digits mix within the
                # same number (found via real output review during Task 1.7's peyk-vlm wiring).
                fragments.append(normalize_digits(markdownify(full_html), lang) if full_html else stub_fragment("peyk-tsr", "table"))
                continue
            structure = tsr_structures.get(idx)
            if structure is None:
                fragments.append(stub_fragment("peyk-tsr", "table"))
            else:
                cell_texts = {
                    cell["cell_i"]: dcr_results.get(f"r{idx}_c{cell['cell_i']}") or ocr_results.get(f"r{idx}_c{cell['cell_i']}") or ""
                    for cell in structure["cells"]
                }
                fragments.append(assemble_markdown_table(structure, cell_texts, lang))
        elif label == "figure":
            region_id = f"r{idx}"
            description = figures_results.get(region_id)
            fragments.append(f"*{description}*" if description else stub_fragment("peyk-vlm", "figure"))
        else:
            print(f"[peyk-orchestrator] unknown region label '{label}', skipping", file=sys.stderr)

    return "\n\n".join(fragments)


def dispatch_documents(docs: list[Path], layout_results: dict[str, dict], config: PipelineConfig, workdir: Path) -> dict[str, str]:
    """Crop every region of every document, dispatch TSR/OCR/figures once each across the whole
    batch (not once per document — see dispatch_tsr_batch/dispatch_ocr_batch/
    dispatch_figures_batch), dispatch DCR per document (deliberately — see dispatch_dcr), and
    assemble each document's markdown. Returns {doc_stem: markdown}, only for documents
    layout_results has regions for."""
    # Created once, upfront, rather than lazily inside dispatch_ocr_batch: prepare_document and
    # process_tsr_results both crop OCR targets directly into it (their final "<doc_stem>__
    # <local_id>.png" name), so it needs to exist before either of them runs.
    ocr_in = workdir / "ocr_in"
    if ocr_in.exists():
        shutil.rmtree(ocr_in)
    ocr_in.mkdir(parents=True, exist_ok=True)

    # Same reasoning as ocr_in above — prepare_document crops figure regions directly into
    # this with their final batch-ready name.
    figures_in = workdir / "figures_in"
    if figures_in.exists():
        shutil.rmtree(figures_in)
    figures_in.mkdir(parents=True, exist_ok=True)

    # No cell_ocr override configured (the common case): table cells share ocr_in/ocr_batch
    # with whole-region crops exactly as before — one combined dispatch, no extra model-load
    # cost. Override configured: cells get their own dir/batch/dispatch, so an isolated cell
    # crop never reaches whatever (possibly VLM-style) model config.ocr uses — see
    # implementation_plan.md Task 1.5.
    cell_ocr_config = config.cell_ocr if config.cell_ocr is not None else config.ocr
    using_separate_cell_ocr = config.cell_ocr is not None
    if using_separate_cell_ocr:
        cell_ocr_in = workdir / "cell_ocr_in"
        if cell_ocr_in.exists():
            shutil.rmtree(cell_ocr_in)
        cell_ocr_in.mkdir(parents=True, exist_ok=True)
        cell_ocr_batch: list[tuple[str, str]] = []
    else:
        cell_ocr_in = ocr_in

    doc_states: dict[str, dict] = {}
    tsr_batch: list[tuple[str, str, Path]] = []
    table_full_batch: list[tuple[str, str, Path]] = []
    ocr_batch: list[tuple[str, str]] = []
    figures_batch: list[tuple[str, str]] = []
    if not using_separate_cell_ocr:
        cell_ocr_batch = ocr_batch
    for doc_path in docs:
        doc_stem = doc_path.stem
        if doc_stem not in layout_results:
            continue
        doc_states[doc_stem] = prepare_document(
            doc_path, layout_results[doc_stem], config, workdir, ocr_in, tsr_batch, table_full_batch, ocr_batch,
            figures_in, figures_batch,
        )

    t0 = time.perf_counter()
    tsr_out = dispatch_tsr_batch(tsr_batch, config, workdir)
    print(f"[peyk-orchestrator] dispatch_tsr_batch (docker run): {time.perf_counter() - t0:.2f}s", file=sys.stderr)

    t0 = time.perf_counter()
    tsr_structures_by_doc = {
        doc_stem: process_tsr_results(doc_state, tsr_out, config, cell_ocr_config, cell_ocr_in, cell_ocr_batch)
        for doc_stem, doc_state in doc_states.items()
    }
    print(f"[peyk-orchestrator] process_tsr_results (all docs, the gap between TSR and OCR): {time.perf_counter() - t0:.2f}s", file=sys.stderr)

    # full_table_backend(config)-matching case only (config.py) — empty otherwise, since
    # prepare_document never populates table_full_batch unless that condition held for the
    # whole run. See dispatch_table_full_batch/assemble_document.
    t0 = time.perf_counter()
    table_full_results_by_doc = dispatch_table_full_batch(table_full_batch, config, workdir)
    print(f"[peyk-orchestrator] dispatch_table_full_batch (docker run): {time.perf_counter() - t0:.2f}s", file=sys.stderr)

    t0 = time.perf_counter()
    ocr_results_by_doc = dispatch_ocr_batch(ocr_batch, config.ocr, workdir, ocr_in)
    print(f"[peyk-orchestrator] dispatch_ocr_batch (docker run): {time.perf_counter() - t0:.2f}s", file=sys.stderr)

    t0 = time.perf_counter()
    figures_results_by_doc = dispatch_figures_batch(figures_batch, config, workdir, figures_in)
    print(f"[peyk-orchestrator] dispatch_figures_batch (docker run): {time.perf_counter() - t0:.2f}s", file=sys.stderr)

    if using_separate_cell_ocr:
        t0 = time.perf_counter()
        cell_ocr_results_by_doc = dispatch_ocr_batch(cell_ocr_batch, cell_ocr_config, workdir, cell_ocr_in, out_dir_name="cell_ocr_out")
        print(f"[peyk-orchestrator] dispatch_cell_ocr_batch (docker run): {time.perf_counter() - t0:.2f}s", file=sys.stderr)
        # Cell ids ("r{idx}_c{cell_i}"/"r{idx}_row{row}") and whole-region ids ("r{idx}") never
        # collide, so merging per doc_stem is safe.
        for doc_stem, cell_results in cell_ocr_results_by_doc.items():
            ocr_results_by_doc.setdefault(doc_stem, {}).update(cell_results)

    markdowns: dict[str, str] = {}
    for doc_path in docs:
        doc_stem = doc_path.stem
        doc_state = doc_states.get(doc_stem)
        if doc_state is None:
            continue
        dcr_results = dispatch_dcr(doc_path, doc_state["dcr_targets"], config, workdir)
        ocr_results = ocr_results_by_doc.get(doc_stem, {})
        table_full_results = table_full_results_by_doc.get(doc_stem, {})
        figures_results = figures_results_by_doc.get(doc_stem, {})
        markdowns[doc_stem] = assemble_document(doc_state, tsr_structures_by_doc[doc_stem], dcr_results, ocr_results, table_full_results, figures_results, config.ocr.lang)
    return markdowns
