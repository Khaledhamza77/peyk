import importlib.util
from pathlib import Path

# Loaded by explicit file path under a private module name, not via `sys.path.insert` + `from
# backends.base import ...` (the convention this stage's other test files use) -- see
# containers/peyk/stages/tsr/tests/test_base.py's own comment on this for the full reasoning.
# col_boxes() here is a near-duplicate of peyk-tsr's, and the two collide under the shared
# generic "backends" module name the moment both stages' test suites run in the same pytest
# session, so this file needs the same collision-proof loader tsr's test uses.
_BASE_PATH = Path(__file__).parent.parent / "backends" / "base.py"
_spec = importlib.util.spec_from_file_location("peyk_surya_backends_base", _BASE_PATH)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
Cell, TableStructure, col_boxes = _base.Cell, _base.TableStructure, _base.col_boxes


def _structure(cells: list[Cell]) -> TableStructure:
    return TableStructure(num_rows=1, num_cols=len(cells), cells=cells)


def test_col_boxes_interior_column_never_inverts():
    # Same adversarial shape as containers/peyk/stages/tsr/tests/test_base.py's equivalent test
    # (this function was ported from that file and carried the identical defect until fixed
    # alongside it): an interior column caught between two garbled neighbors, where clamping a
    # boundary against only its immediate neighbor still inverts it two columns later. Confirmed
    # this reproduces here too before the fix, with the same inverted values as the tsr copy.
    structure = _structure([
        Cell(row=0, col=0, row_span=1, col_span=1, bbox=(0.0, 0.0, 100.0, 50.0)),
        Cell(row=0, col=1, row_span=1, col_span=1, bbox=(300.0, 0.0, 310.0, 50.0)),
        Cell(row=0, col=2, row_span=1, col_span=1, bbox=(50.0, 0.0, 60.0, 50.0)),
    ])
    cols = col_boxes(structure, image_width=310.0, image_height=50.0)
    assert len(cols) == 3
    for c in cols:
        x0, _, x1, _ = c.bbox
        assert x1 >= x0, f"col {c.col} inverted: {c.bbox}"
    boundaries = [cols[0].bbox[0]] + [c.bbox[2] for c in cols]
    assert boundaries == sorted(boundaries)


def test_col_boxes_ordinary_case_matches_plain_midpoint_math():
    structure = _structure([
        Cell(row=0, col=0, row_span=1, col_span=1, bbox=(0.0, 0.0, 100.0, 50.0)),
        Cell(row=0, col=1, row_span=1, col_span=1, bbox=(120.0, 0.0, 200.0, 50.0)),
    ])
    cols = col_boxes(structure, image_width=200.0, image_height=50.0)
    assert cols[0].bbox == (0.0, 0.0, 110.0, 50.0)
    assert cols[1].bbox == (110.0, 0.0, 200.0, 50.0)
