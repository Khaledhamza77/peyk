from pathlib import Path

from .base import Cell, TSRBackend, TableStructure


class RapidTableBackend(TSRBackend):
    name = "rapidtable"

    # rapid_table.utils.ModelType.SLANETPLUS (no underscore before "PLUS" — confirmed against
    # the installed package; other members are PPSTRUCTURE_EN/PPSTRUCTURE_ZH/UNITABLE). Its
    # own value is the string "slanet_plus", but RapidTableInput requires the actual Enum
    # member, not that string (AttributeError: 'str' object has no attribute 'value') — same
    # gotcha as rapidocr's ModelType/LangRec/OCRVersion params in peyk-simple-ocr.
    def __init__(self, model_type: str = "SLANETPLUS"):
        self.model_type = model_type
        self._engine = None

    def load(self) -> None:
        from rapid_table import RapidTable, RapidTableInput
        from rapid_table.utils import ModelType

        # use_ocr=False: without it, RapidTable.__call__ always tries to run its own internal
        # OCR engine to fill in cell text (even with no ocr_results passed in) and crashes if
        # one isn't installed/configured (TypeError: 'NoneType' object is not callable, from
        # get_ocr_results calling self.ocr_engine). With it False, `__call__` skips OCR
        # entirely and returns just the structure model's own cell_bboxes/logic_points, which
        # is all this backend needs — peyk-orchestrator pairs cell text in separately.
        self._engine = RapidTable(RapidTableInput(model_type=ModelType[self.model_type], use_ocr=False))

    def predict(self, image_path: Path) -> TableStructure:
        if self._engine is None:
            raise RuntimeError("Backend not loaded; call load() first")

        result = self._engine(str(image_path))

        # cell_bboxes/logic_points are batched — one array per input image, since RapidTable's
        # __call__ accepts a list of images (see BatchRec logging). Only ever predicting one
        # crop at a time here, so always take the first (only) image's arrays; each row within
        # it is one cell (confirmed shapes: cell_bboxes[0] is (num_cells, 8),
        # logic_points[0] is (num_cells, 4) — not a flat per-cell list at the top level).
        # `logic_points[i]` = [row_start, row_end, col_start, col_end] for `cell_bboxes[i]`
        # (RapidTable's own pairing, by shared index). `cell_bboxes[i]` is an 8-value quad
        # (x1,y1,x2,y2,x3,y3,x4,y4); reduced to an axis-aligned bbox since the rest of the
        # pipeline (crop_region, peyk-dcr) is axis-aligned-bbox-only throughout.
        cells: list[Cell] = []
        num_rows = num_cols = 0
        for quad, logic in zip(result.cell_bboxes[0], result.logic_points[0]):
            xs = quad[0::2]
            ys = quad[1::2]
            row_start, row_end, col_start, col_end = (int(v) for v in logic)
            cells.append(
                Cell(
                    row=row_start,
                    col=col_start,
                    row_span=row_end - row_start + 1,
                    col_span=col_end - col_start + 1,
                    bbox=(float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))),
                )
            )
            num_rows = max(num_rows, row_end + 1)
            num_cols = max(num_cols, col_end + 1)

        return TableStructure(num_rows=num_rows, num_cols=num_cols, cells=cells)
