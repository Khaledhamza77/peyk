import importlib.util
from pathlib import Path

# Every stage under containers/peyk/stages/ names its backend package the same generic
# "backends" (see surya/tsr/layout/ocr/vlm's own tests/run.py, all `sys.path.insert` + `from
# backends.X import Y`). That is harmless in production -- each container only ever has its own
# stage on sys.path -- but it collides the moment two stages' test suites run in the same pytest
# session: Python caches modules by name, so whichever stage's tests import `backends` first
# "wins", and every other stage's `from backends.base import ...` silently resolves against that
# first stage's package instead of its own. Confirmed concretely: running this file alongside
# surya's test suite loaded surya's OWN (structurally identical but independently-tracked)
# col_boxes() under the name `backends.base`, so a test meant to exercise the fix here appeared
# to pass or fail based on surya's copy, not this file's. Loading by explicit file path under a
# private name sidesteps the collision entirely, regardless of what any other stage's tests
# already imported this session.
_BASE_PATH = Path(__file__).parent.parent / "backends" / "base.py"
_spec = importlib.util.spec_from_file_location("peyk_tsr_backends_base", _BASE_PATH)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
Cell, TableStructure, col_boxes = _base.Cell, _base.TableStructure, _base.col_boxes


def _structure(cells: list[Cell]) -> TableStructure:
    return TableStructure(num_rows=1, num_cols=len(cells), cells=cells)


def test_col_boxes_ordinary_case_matches_plain_midpoint_math():
    # Two columns, already left-to-right ordered with a small gap between them -- the
    # running-maximum boundary construction must reduce to the plain midpoint here, identical
    # to what the original (pre-fix) code produced for any already-well-ordered input.
    structure = _structure([
        Cell(row=0, col=0, row_span=1, col_span=1, bbox=(0.0, 0.0, 100.0, 50.0)),
        Cell(row=0, col=1, row_span=1, col_span=1, bbox=(120.0, 0.0, 200.0, 50.0)),
    ])
    cols = col_boxes(structure, image_width=200.0, image_height=50.0)
    assert [c.col for c in cols] == [0, 1]
    assert cols[0].bbox == (0.0, 0.0, 110.0, 50.0)  # midpoint of 100 and 120
    assert cols[1].bbox == (110.0, 0.0, 200.0, 50.0)
    for c in cols:
        x0, _, x1, _ = c.bbox
        assert x1 >= x0


def test_col_boxes_out_of_order_neighbor_never_inverts():
    # column 1's raw x-range sits entirely left of column 0's. Confirmed by direct measurement
    # against the code before this fix (a 2-column version of this case): the midpoint
    # (110+0)/2=55 falls left of col0's own x0 (100), inverting it to (100.0, ..., 55.0, ...).
    structure = _structure([
        Cell(row=0, col=0, row_span=1, col_span=1, bbox=(100.0, 0.0, 110.0, 50.0)),
        Cell(row=0, col=1, row_span=1, col_span=1, bbox=(0.0, 0.0, 10.0, 50.0)),
    ])
    cols = col_boxes(structure, image_width=150.0, image_height=50.0)
    for c in cols:
        x0, _, x1, _ = c.bbox
        assert x1 >= x0, f"col {c.col} inverted: {c.bbox}"


def test_col_boxes_interior_column_never_inverts_even_when_pairwise_clamping_would_fail():
    # This is the case that actually crashed in production (fawry_sample.pdf's r13 region under
    # layout=doclayout-yolo + tsr=tableformer): an INTERIOR column caught between two garbled
    # neighbors, where a naive per-neighbor clamp still fails. Confirmed by direct measurement:
    # with a first-draft fix that clamped each boundary only against its own immediate neighbor,
    # this exact input still produced col 1 = (200.0, 0.0, 60.0, 50.0) -- inverted -- because col
    # 0's already-advanced right edge (200, fixed by the first stitching step) collided with col
    # 2's raw left edge (60, still unadjusted), and clamping against col 2 alone pulled the
    # boundary below col 0's already-fixed edge. Only a running-maximum construction over the
    # whole boundary sequence (not a per-step clamp) closes this.
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
    # Boundaries must also be non-decreasing left to right -- the actual invariant the fix
    # provides, stronger than checking each span individually.
    boundaries = [cols[0].bbox[0]] + [c.bbox[2] for c in cols]
    assert boundaries == sorted(boundaries)


def test_col_boxes_overlapping_columns_never_invert():
    # A milder degenerate case than full reversal: the columns overlap rather than swap order
    # outright. Confirmed this one was already fine even before the fix (the running-max
    # construction changes nothing here) -- kept as a case the fix must not regress.
    structure = _structure([
        Cell(row=0, col=0, row_span=1, col_span=1, bbox=(0.0, 0.0, 100.0, 50.0)),
        Cell(row=0, col=1, row_span=1, col_span=1, bbox=(60.0, 0.0, 160.0, 50.0)),
        Cell(row=0, col=2, row_span=1, col_span=1, bbox=(150.0, 0.0, 220.0, 50.0)),
    ])
    cols = col_boxes(structure, image_width=220.0, image_height=50.0)
    assert len(cols) == 3
    for c in cols:
        x0, _, x1, _ = c.bbox
        assert x1 >= x0, f"col {c.col} inverted: {c.bbox}"


def test_col_boxes_single_column_spans_full_width():
    structure = _structure([
        Cell(row=0, col=0, row_span=1, col_span=1, bbox=(10.0, 0.0, 90.0, 50.0)),
    ])
    cols = col_boxes(structure, image_width=100.0, image_height=50.0)
    assert cols[0].bbox == (0.0, 0.0, 100.0, 50.0)
