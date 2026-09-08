"""Shared dataclasses + cell-grid calibration helpers, matching the exact output contracts
peyk-orchestrator already expects from peyk-layout (Region), peyk-tsr (Cell/TableStructure +
row_boxes/col_boxes/regularized_cells), and peyk-simple-ocr/peyk-paddleocr-vl (OCRResult).

Deliberately duplicated rather than imported across containers (same reasoning RENDER_SCALE
is already duplicated in every container that renders pages) — each container stays buildable
and deployable independently, with no cross-container Python import path to keep in sync."""
from dataclasses import dataclass


@dataclass
class Region:
    """A detected layout region on a page. Matches peyk-layout/backends/base.py's Region
    exactly — see that file for field meaning."""

    page: int
    label: str  # one of: text, table, figure
    score: float
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1 in pixel coords

    def to_dict(self) -> dict:
        return {
            "page": self.page,
            "label": self.label,
            "score": self.score,
            "bbox": list(self.bbox),
        }


@dataclass
class Cell:
    """One table cell, in the grid coordinate system (row 0 / col 0 = top-left). Matches
    peyk-tsr/backends/base.py's Cell exactly."""

    row: int
    col: int
    row_span: int
    col_span: int
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1, pixel coords local to the input crop

    def to_dict(self) -> dict:
        return {
            "row": self.row,
            "col": self.col,
            "row_span": self.row_span,
            "col_span": self.col_span,
            "bbox": list(self.bbox),
        }


@dataclass
class TableStructure:
    num_rows: int
    num_cols: int
    cells: list[Cell]

    def to_dict(self) -> dict:
        return {
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "cells": [c.to_dict() for c in self.cells],
        }


@dataclass
class OCRResult:
    """Recognized text for a single text-region crop. Matches
    peyk-simple-ocr/backends/base.py's OCRResult exactly."""

    text: str
    score: float

    def to_dict(self) -> dict:
        return {"text": self.text, "score": self.score}


@dataclass
class RowBox:
    row: int
    bbox: tuple[float, float, float, float]

    def to_dict(self) -> dict:
        return {"row": self.row, "bbox": list(self.bbox)}


def row_boxes(structure: TableStructure, image_width: float) -> list[RowBox]:
    """Ported verbatim from peyk-tsr/backends/base.py — see that file for the full rationale
    (full-crop-width row bands calibrated only from the model's row-height signal)."""
    rows: dict[int, list[tuple[float, float, float, float]]] = {}
    for cell in structure.cells:
        for r in range(cell.row, cell.row + cell.row_span):
            rows.setdefault(r, []).append(cell.bbox)

    result = []
    for r in sorted(rows):
        boxes = rows[r]
        y0 = min(b[1] for b in boxes)
        y1 = max(b[3] for b in boxes)
        result.append(RowBox(row=r, bbox=(0.0, y0, image_width, y1)))
    return result


@dataclass
class ColBox:
    col: int
    bbox: tuple[float, float, float, float]

    def to_dict(self) -> dict:
        return {"col": self.col, "bbox": list(self.bbox)}


def col_boxes(structure: TableStructure, image_width: float, image_height: float) -> list[ColBox]:
    """Ported from peyk-tsr/backends/base.py — see that file for the full rationale
    (full-crop-height column bands, boundaries extended to the midpoint with each neighbor so
    every x-position belongs to exactly one column) and for why the boundary sequence is built
    as a running maximum rather than each span's edges being clamped pairwise in place: a
    noisy/garbled structure prediction can put an interior column's raw x-range so far out of
    index order that a pairwise clamp against only its immediate neighbor still inverts it once
    an already-adjusted edge collides with a raw edge two columns down the line. Not just a
    theoretical case here either — this path has no `--visualize` guard at all (unlike
    peyk-tsr's draw-time crash, an inverted box here would silently corrupt regularized_cells()'s
    real OCR crop regions on genuinely noisy/scanned-table predictions, exactly the input this
    stage's own smart-table-split feature exists to handle, with no crash to surface it."""
    cols: dict[int, list[tuple[float, float, float, float]]] = {}
    for cell in structure.cells:
        for c in range(cell.col, cell.col + cell.col_span):
            cols.setdefault(c, []).append(cell.bbox)

    indices = sorted(cols)
    spans = []
    for c in indices:
        boxes = cols[c]
        spans.append([min(b[0] for b in boxes), max(b[2] for b in boxes)])

    boundaries = [0.0]
    for i in range(len(spans) - 1):
        midpoint = (spans[i][1] + spans[i + 1][0]) / 2
        boundaries.append(max(midpoint, boundaries[-1]))
    boundaries.append(image_width)

    return [
        ColBox(col=c, bbox=(boundaries[i], 0.0, boundaries[i + 1], image_height))
        for i, c in enumerate(indices)
    ]


def regularized_cells(structure: TableStructure, rows: list[RowBox], cols: list[ColBox]) -> list[dict]:
    """Ported verbatim from peyk-tsr/backends/base.py — row-band x column-band intersection,
    replacing each cell's raw model bbox. row_span/col_span carried into the output so a
    consumer of just this "_aug.json" file can tell whether a cell is merged (e.g. to avoid
    cutting a table-split boundary through it) without also loading the raw structure JSON."""
    row_bbox_by_idx = {r.row: r.bbox for r in rows}
    col_bbox_by_idx = {c.col: c.bbox for c in cols}

    result = []
    for cell in structure.cells:
        row_range = range(cell.row, cell.row + cell.row_span)
        col_range = range(cell.col, cell.col + cell.col_span)
        ys = [row_bbox_by_idx[r][i] for r in row_range if r in row_bbox_by_idx for i in (1, 3)]
        xs = [col_bbox_by_idx[c][i] for c in col_range if c in col_bbox_by_idx for i in (0, 2)]
        if not ys or not xs:
            continue
        result.append({
            "row": cell.row,
            "col": cell.col,
            "row_span": cell.row_span,
            "col_span": cell.col_span,
            "bbox": [min(xs), min(ys), max(xs), max(ys)],
        })
    return result
