"""Counts the units of work one pipeline run actually performed, so latency can be normalised.

Raw seconds-per-document is close to meaningless across this project's own sample set: a 3-page
document carrying one 120-cell table costs far more than a 20-page prose document, and the two
numbers are not comparable. Every headline latency figure should be divided by one of these
counts -- seconds per page, per region crop, per table cell -- and which denominator is the right
one depends on the stage.

Counts come from the shared peyk-hotstorage-workdir named volume, read via a single throwaway
`alpine find` container rather than PeykRunner.mirror_workdir_to_host(). A real run's table-cell
crops alone can be hundreds of files, and this needs only their names, never their contents --
copying the whole volume to the host to count filenames would dominate the benchmark's own
runtime.

Filename conventions this parses are pipeline.py's own (see dispatch_documents/
process_tsr_results):
    layout_out/<doc>_p<N>_raw.png     one per rendered page
    layout_out/<doc>.json             detected regions for that document
    ocr_in/<doc>__r<N>.png            one whole-region text crop
    cell_ocr_in/<doc>__r<N>_c<M>.png  one table cell crop  (also r<N>_row<M> in row mode)
    tsr_in/<doc>__r<N>.png            one table region
    figures_in/<doc>__r<N>.png        one figure region
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import docker

WORKDIR_VOLUME = "peyk-hotstorage-workdir"

# Matches a table-cell or table-row crop id, which is what separates a per-cell work unit from a
# whole-region one. pipeline.py emits "r12_c0" for cell mode and "r12_row3" for row mode; a plain
# "r12" is a whole region.
_CELL_ID = re.compile(r"__r\d+_(?:c|row)\d+\.")
_PAGE_RENDER = re.compile(r"_p(\d+)_raw\.png$")


@dataclass
class WorkUnits:
    """Every count is "files this run left in the workdir", not "files the model succeeded on" --
    a failed OCR call still leaves its input crop behind. That is the correct denominator for
    latency (the work was dispatched and paid for either way), and the wrong one for any success
    rate, which is what the stub/error events in the JobStore are for."""

    pages: int = 0
    documents: int = 0
    region_crops: int = 0
    table_regions: int = 0
    table_cells: int = 0
    figure_regions: int = 0
    fullpage_images: int = 0
    per_dir_file_counts: dict[str, int] = field(default_factory=dict)

    def denominator(self, stage: str, stages_present: set[str] | None = None) -> tuple[str, int] | None:
        """The unit a given stage's latency should be divided by. Returns None when the stage has
        no sensible denominator or nothing was dispatched to it, so callers report a bare total
        rather than dividing by zero.

        `stages_present` matters for one real case. pipeline.py only gives table cells their own
        dispatch when `cell_ocr` differs from `ocr`; when the two resolve to the same StageConfig
        it merges them into a single `ocr` dispatch to avoid loading the same local model twice.
        In that merged case there is no `cell_ocr` event at all, and the `ocr` stage's time covers
        whole-region crops AND every table cell -- so dividing it by region crops alone overstates
        per-crop cost by however many cells were folded in (measured: 7 crops vs 364 cells on a
        single sample document, a ~50x error). Pass the stages that actually ran so the
        denominator can account for it."""
        merged_cells = (
            stage == "ocr"
            and stages_present is not None
            and "cell_ocr" not in stages_present
            and self.table_cells > 0
        )
        if merged_cells:
            return ("text unit (crops+cells)", self.region_crops + self.table_cells)

        mapping = {
            "layout": ("page", self.pages),
            "ocr": ("region crop", self.region_crops),
            "cell_ocr": ("table cell", self.table_cells),
            "tsr": ("table", self.table_regions),
            "table_full": ("table", self.table_regions),
            "figures": ("figure", self.figure_regions),
            "dcr": ("document", self.documents),
            "fullpage": ("page", self.fullpage_images or self.pages),
        }
        got = mapping.get(stage)
        if got is None or got[1] <= 0:
            return None
        return got

    def to_dict(self) -> dict:
        return {
            "pages": self.pages,
            "documents": self.documents,
            "region_crops": self.region_crops,
            "table_regions": self.table_regions,
            "table_cells": self.table_cells,
            "figure_regions": self.figure_regions,
            "fullpage_images": self.fullpage_images,
            "per_dir_file_counts": self.per_dir_file_counts,
        }


def _list_workdir_files(client: "docker.DockerClient", volume: str = WORKDIR_VOLUME) -> list[str]:
    """One `alpine find` container, newline-separated absolute paths under /w. Returns [] rather
    than raising if the volume does not exist yet (nothing has run) -- an empty count set is the
    honest answer there, not an error."""
    try:
        raw = client.containers.run(
            "alpine",
            command=["find", "/w", "-type", "f"],
            volumes={volume: {"bind": "/w", "mode": "ro"}},
            remove=True,
            stdout=True,
            stderr=False,
        )
    except docker.errors.DockerException:
        return []
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
    return [line.strip() for line in text.splitlines() if line.strip()]


def count(
    client: "docker.DockerClient",
    volume: str = WORKDIR_VOLUME,
    only_dirs: set[str] | None = None,
) -> WorkUnits:
    """`only_dirs` restricts counting to those top-level workdir directory names.

    Pass it whenever counting a specific job. The workdir volume is persistent by design, so it
    accumulates directories from earlier runs under different configs; counting the whole volume
    silently folds those into the current run's totals. harness.Harness._work_units() derives the
    set from the job's own dispatch events, which is the only authoritative record of what a given
    run actually touched. Counting everything (None) is correct only for an ad-hoc look at the
    volume, never for a benchmark record."""
    paths = _list_workdir_files(client, volume)
    units = WorkUnits()
    documents: set[str] = set()

    for path in paths:
        parts = path.split("/")
        # /w/<stage_dir>/... -- anything shallower is a stray file at the volume root.
        if len(parts) < 4:
            continue
        stage_dir = parts[2]
        if only_dirs is not None and stage_dir not in only_dirs:
            continue
        name = parts[-1]
        units.per_dir_file_counts[stage_dir] = units.per_dir_file_counts.get(stage_dir, 0) + 1

        if stage_dir == "layout_out":
            if _PAGE_RENDER.search(name):
                units.pages += 1
            elif name.endswith(".json"):
                documents.add(name[: -len(".json")])
        elif stage_dir == "ocr_in" and name.endswith(".png"):
            # Without a configured cell_ocr override, pipeline.py puts cell crops in ocr_in too
            # (dispatch_documents shares one batch) -- so split them by id shape here rather than
            # assuming the presence of a cell_ocr_in directory.
            if _CELL_ID.search(name):
                units.table_cells += 1
            else:
                units.region_crops += 1
        elif stage_dir == "cell_ocr_in" and name.endswith(".png"):
            units.table_cells += 1
        elif stage_dir == "tsr_in" and name.endswith(".png"):
            units.table_regions += 1
        elif stage_dir == "figures_in" and name.endswith(".png"):
            units.figure_regions += 1
        elif stage_dir in ("fullpage_in", "vlm_fullpage_in") and name.endswith(".png"):
            units.fullpage_images += 1

    units.documents = len(documents)
    return units
