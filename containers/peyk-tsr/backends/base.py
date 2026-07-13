import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Cell:
    """One table cell, in the grid coordinate system (row 0 / col 0 = top-left)."""

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
class RowBox:
    """One row's bbox, spanning the full crop width: x0=0/x1=image_width always, y0/y1 taken
    from the cells occupying that row (including cells whose row_span extends into it from an
    earlier row). A safety net for downstream text pairing when per-cell boxes clip or miss
    real content (e.g. a header row whose individual cell boxes came out too narrow) — using
    the full crop width rather than the union of cell x-bounds means this only depends on the
    model getting the row's vertical extent roughly right, not any individual cell's
    horizontal extent. In effect, the structure model's job for this artifact reduces to
    calibrating row height boundaries; column information isn't part of it (that's still
    what the per-cell boxes are for)."""

    row: int
    bbox: tuple[float, float, float, float]

    def to_dict(self) -> dict:
        return {"row": self.row, "bbox": list(self.bbox)}


def row_boxes(structure: TableStructure, image_width: float) -> list[RowBox]:
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
    """Column counterpart to RowBox: y0=0/y1=image_height always, x0/x1 taken from the cells
    occupying that column (including cells whose col_span extends into it). Calibrates column
    width the same way RowBox calibrates row height — trusts the model's x-extent for this
    column, not its y-extent (that's RowBox's job)."""

    col: int
    bbox: tuple[float, float, float, float]

    def to_dict(self) -> dict:
        return {"col": self.col, "bbox": list(self.bbox)}


def col_boxes(structure: TableStructure, image_width: float, image_height: float) -> list[ColBox]:
    cols: dict[int, list[tuple[float, float, float, float]]] = {}
    for cell in structure.cells:
        for c in range(cell.col, cell.col + cell.col_span):
            cols.setdefault(c, []).append(cell.bbox)

    indices = sorted(cols)
    spans = []  # [x0, x1] per column, before the contiguity pass below
    for c in indices:
        boxes = cols[c]
        spans.append([min(b[0] for b in boxes), max(b[2] for b in boxes)])

    # Extend each boundary to the midpoint with its neighbor, so adjacent columns always
    # meet with no gap and no overlap — any content sitting between two detected column
    # edges (which the raw per-column union above would otherwise miss) ends up on one side
    # or the other instead of belonging to neither. First/last columns extend all the way to
    # the crop edges, same reasoning row_boxes uses for the full row-band width.
    for i in range(len(spans) - 1):
        boundary = (spans[i][1] + spans[i + 1][0]) / 2
        spans[i][1] = boundary
        spans[i + 1][0] = boundary
    if spans:
        spans[0][0] = 0.0
        spans[-1][1] = image_width

    return [ColBox(col=c, bbox=(x0, 0.0, x1, image_height)) for c, (x0, x1) in zip(indices, spans)]


def regularized_cells(structure: TableStructure, rows: list[RowBox], cols: list[ColBox]) -> list[dict]:
    """One bbox per original cell, replacing the model's raw (possibly noisy) per-cell box
    with the intersection of that cell's row band (height, from RowBox) and column band
    (width, from ColBox) — i.e. a calibrated grid built entirely from the more reliable
    row-height/column-width signals, rather than trusting any individual cell detection.
    Spanning cells union the row/column bands they cover. Used for OCR crop regularization
    (pipeline.py) rather than replacing the raw cells list wholesale, since markdown assembly
    still needs the model's own row/col indices — this only replaces the bbox."""
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
        result.append({"row": cell.row, "col": cell.col, "bbox": [min(xs), min(ys), max(xs), max(ys)]})
    return result


class TSRBackend(ABC):
    """Common interface every table-structure-recognition backend must implement.
    Structure only, no text: peyk-orchestrator pairs cell bboxes with peyk-dcr (born-digital)
    or the OCR containers (scanned) itself, rather than this container calling out to either —
    see pipeline.md / implementation_plan.md Task 1.5."""

    name: str

    @abstractmethod
    def load(self) -> None:
        """Load model weights. Called once before predict()."""

    @abstractmethod
    def predict(self, image_path: Path) -> TableStructure:
        """Run table structure recognition on a single table-region crop."""


def grid_from_rows_and_cols(
    row_bboxes: list[tuple[float, float, float, float]],
    col_bboxes: list[tuple[float, float, float, float]],
) -> TableStructure:
    """Build a cell grid from independently-detected row and column bands (TATR's output
    shape: separate 'table row'/'table column' boxes, not per-cell boxes). Each cell is the
    column's x-range crossed with the row's y-range — not a bbox intersection, since a row
    box spans the full table width and a column box spans the full table height, so a literal
    intersection would just return the (tiny or zero) overlap between two thin strips.

    Simplification: every cell is emitted as row_span=col_span=1 — merged/spanning cells
    (TATR's 'table spanning cell'/'table projected row header' classes) are not reconstructed
    into multi-cell spans. Adequate for the PoC's markdown-table output; revisit if spanning
    headers show up often enough in real documents to garble the assembled table.
    """
    rows_sorted = sorted(row_bboxes, key=lambda b: (b[1] + b[3]) / 2)
    cols_sorted = sorted(col_bboxes, key=lambda b: (b[0] + b[2]) / 2)
    cells = [
        Cell(
            row=r_idx,
            col=c_idx,
            row_span=1,
            col_span=1,
            bbox=(col[0], row[1], col[2], row[3]),
        )
        for r_idx, row in enumerate(rows_sorted)
        for c_idx, col in enumerate(cols_sorted)
    ]
    return TableStructure(num_rows=len(rows_sorted), num_cols=len(cols_sorted), cells=cells)


def grid_from_html_tokens(structure_tokens: list[str], cell_bboxes: list[list[float]]) -> TableStructure:
    """Shared by the two backends (pp_structure.py, tableformer.py) whose underlying models
    predict an HTML-token stream (`<tr>`, `<td`, ` colspan="2"`, `>`, `</td>`, ...) plus one
    bbox per opened `<td`, in reading order — TableFormer and PaddleOCR/PaddleX's table
    modules share this same output paradigm. Walks the tokens rebuilding row/col position,
    respecting spans already claimed by an earlier row/col so a rowspan correctly pushes a
    later row's column index over, and pairs each `<td` with the next bbox in the list."""
    cells: list[Cell] = []
    occupied: set[tuple[int, int]] = set()
    row = -1
    col = 0
    bbox_iter = iter(cell_bboxes)

    i = 0
    while i < len(structure_tokens):
        tok = structure_tokens[i]
        if tok == "<tr>":
            row += 1
            col = 0
        elif tok.startswith("<td"):
            attrs = tok
            while ">" not in attrs and i + 1 < len(structure_tokens):
                i += 1
                attrs += structure_tokens[i]
            colspan_m = re.search(r'colspan="(\d+)"', attrs)
            rowspan_m = re.search(r'rowspan="(\d+)"', attrs)
            colspan = int(colspan_m.group(1)) if colspan_m else 1
            rowspan = int(rowspan_m.group(1)) if rowspan_m else 1

            while (row, col) in occupied:
                col += 1

            bbox = next(bbox_iter, None)
            if bbox is not None:
                xs, ys = bbox[0::2], bbox[1::2]
                cells.append(
                    Cell(
                        row=row,
                        col=col,
                        row_span=rowspan,
                        col_span=colspan,
                        bbox=(min(xs), min(ys), max(xs), max(ys)),
                    )
                )
            for rr in range(row, row + rowspan):
                for cc in range(col, col + colspan):
                    occupied.add((rr, cc))
            col += colspan
        i += 1

    num_rows = max((c.row + c.row_span for c in cells), default=0)
    num_cols = max((c.col + c.col_span for c in cells), default=0)
    return TableStructure(num_rows=num_rows, num_cols=num_cols, cells=cells)
