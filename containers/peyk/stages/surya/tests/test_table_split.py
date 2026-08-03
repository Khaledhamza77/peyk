import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image

from backends.table_split import (
    body_center,
    center_split_index,
    compute_chunks,
    max_per_chunk_to_boundaries,
    split_cols,
    split_rows,
)


def _row_bands(n, row_height=100):
    """n body rows (row indices 1..n), evenly stacked, each row_height tall, full width 1000."""
    return [{"row": i, "bbox": [0, (i - 1) * row_height, 1000, i * row_height]} for i in range(1, n + 1)]


def test_body_center_uses_only_the_bands_own_extent():
    # 5 rows starting at y=0, each 100px -> extent is y=0..500, center=250
    bands = _row_bands(5)
    assert body_center(bands, axis_key="row") == 250.0


def test_center_split_index_picks_the_boundary_nearest_center():
    bands = _row_bands(5)  # boundaries at y=100,200,300,400; center=250 -> nearest is y=200 (k=2) or y=300 (k=3), tie goes to first found (k=2)
    k = center_split_index(bands, body_center(bands, axis_key="row"), axis_key="row")
    assert k == 2
    assert bands[k - 1]["row"] == 2 and bands[k]["row"] == 3


def test_center_split_index_avoids_a_merged_cell_straddling_the_natural_boundary():
    bands = _row_bands(5)
    # A cell spanning rows 2-3 (row_span=2) straddles the k=2 boundary picked above.
    cells = [{"row": 2, "col": 0, "row_span": 2, "col_span": 1}]
    k = center_split_index(bands, body_center(bands, axis_key="row"), axis_key="row", cells=cells)
    assert k != 2
    # k=3 (between row 3 and row 4) is the next-nearest boundary and is safe.
    assert k == 3


def test_center_split_index_raises_if_every_boundary_is_unsafe():
    bands = _row_bands(3)
    # One cell spans all 3 rows -> every possible cut (k=1, k=2) straddles it.
    cells = [{"row": 1, "col": 0, "row_span": 3, "col_span": 1}]
    try:
        center_split_index(bands, body_center(bands, axis_key="row"), axis_key="row", cells=cells)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_max_per_chunk_to_boundaries_without_span_awareness():
    assert max_per_chunk_to_boundaries(8, 3) == [3, 6]


def test_max_per_chunk_to_boundaries_nudges_an_unsafe_boundary():
    bands = _row_bands(8)
    cells = [{"row": 3, "col": 0, "row_span": 2, "col_span": 1}]  # spans rows 3-4, straddles k=3
    boundaries = max_per_chunk_to_boundaries(8, 3, bands=bands, cells=cells, axis_key="row")
    assert 3 not in boundaries
    assert boundaries == [2, 6]


def test_compute_chunks_splits_into_the_right_groups():
    bands = _row_bands(5)
    chunks = compute_chunks(bands, [2], axis_key="row")
    assert len(chunks) == 2
    _, _, first_indices = chunks[0]
    _, _, second_indices = chunks[1]
    assert first_indices == [1, 2]
    assert second_indices == [3, 4, 5]


def test_compute_chunks_cut_lands_at_mid_gap_not_inside_a_band():
    bands = _row_bands(5)  # row1 y=[0,100], row2 y=[100,200], row3 y=[200,300], ...
    chunks = compute_chunks(bands, [2], axis_key="row")
    first_span0, first_span1, _ = chunks[0]
    second_span0, second_span1, _ = chunks[1]
    assert first_span0 == 0
    assert first_span1 == second_span0 == 200  # midpoint between row2's bottom (200) and row3's top (200)
    assert second_span1 is None  # last chunk crops to image edge


def test_split_rows_produces_one_image_per_chunk_with_full_width():
    image = Image.new("RGB", (1000, 500), "white")
    rows = [{"row": 0, "bbox": [0, 0, 1000, 50]}] + _row_bands(5)[:5]  # row 0 = header band
    # shift body rows down to leave room for the header band at y=0..50
    rows = [{"row": 0, "bbox": [0, 0, 1000, 50]}] + [
        {"row": i, "bbox": [0, 50 + (i - 1) * 90, 1000, 50 + i * 90]} for i in range(1, 6)
    ]
    body = rows[1:]
    chunks = split_rows(image, rows, [2])
    assert len(chunks) == 2
    for chunk_image, _ in chunks:
        assert chunk_image.width == 1000


def test_split_rows_first_chunk_does_not_duplicate_the_header_band():
    # compute_chunks returns span0=0 (absolute image top) for the first chunk — correct for
    # split_cols (no header), but for split_rows that's the header's own region, which is
    # already pasted separately. A single chunk (no internal boundary) must capture the header
    # once and the body once: total height == original image height, not header+full-image.
    h = 500
    image = Image.new("RGB", (1000, h), "white")
    rows = [{"row": 0, "bbox": [0, 0, 1000, 50]}] + [
        {"row": i, "bbox": [0, 50 + (i - 1) * 90, 1000, 50 + i * 90]} for i in range(1, 6)
    ]
    chunks = split_rows(image, rows, [])
    assert len(chunks) == 1
    combined, _ = chunks[0]
    assert combined.height == h


def test_split_cols_left_to_right_pixel_order():
    image = Image.new("RGB", (1000, 200), "white")
    cols = [{"col": i, "bbox": [i * 200, 0, (i + 1) * 200, 200]} for i in range(5)]
    chunks = split_cols(image, cols, [2])
    assert len(chunks) == 2
    _, first_indices = chunks[0]
    _, second_indices = chunks[1]
    assert first_indices == [0, 1]
    assert second_indices == [2, 3, 4]
    # chunk 0 is the pixel-LEFT crop
    assert chunks[0][0].width < image.width
