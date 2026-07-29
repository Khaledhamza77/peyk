import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backends.table_stitch import (
    detect_label_chunk_index,
    parse_html_table,
    render_html,
    stitch_landscape,
    stitch_portrait,
)


def test_parse_html_table_strips_bold_and_span_tags():
    html = "<table><tbody><tr><th>H1</th><th>H2</th></tr><tr><td><b>row1</b></td><td><span></span></td></tr></tbody></table>"
    grid = parse_html_table(html)
    assert grid == [["H1", "H2"], ["row1", ""]]


def test_stitch_portrait_drops_every_chunks_own_header_except_the_first():
    chunk0 = [["H1", "H2"], ["a", "1"], ["b", "2"]]
    chunk1 = [["H1", "H2"], ["c", "3"]]  # chunk1's own re-recognized header — must be dropped
    final_rows, warnings = stitch_portrait([chunk0, chunk1], expected_row_counts=[2, 1])
    assert final_rows == [["H1", "H2"], ["a", "1"], ["b", "2"], ["c", "3"]]
    assert warnings == []


def test_stitch_portrait_recovers_one_fully_empty_surplus_row():
    chunk0 = [["H1", "H2"], ["a", "1"], ["", ""], ["b", "2"]]  # one genuinely blank row, expected 2
    final_rows, warnings = stitch_portrait([chunk0], expected_row_counts=[2])
    assert final_rows == [["H1", "H2"], ["a", "1"], ["b", "2"]]
    assert any("dropped 1 fully-empty row" in w for w in warnings)


def test_stitch_portrait_does_not_drop_a_row_of_dashes():
    # "-" is legitimate financial content (no entry), never treated as the empty-row artifact.
    chunk0 = [["H1", "H2"], ["a", "1"], ["b", "-"], ["c", "2"]]
    final_rows, warnings = stitch_portrait([chunk0], expected_row_counts=[2])
    assert len(final_rows) == 4  # nothing dropped
    assert any("expected 2 body rows, got 3" in w for w in warnings)


def test_stitch_landscape_joins_by_row_index_in_the_given_chunk_order():
    left = [["H1", "H2"], ["a", "1"]]
    right = [["H3", "H4"], ["b", "2"]]
    final_rows, warnings = stitch_landscape([left, right], expected_col_counts=[2, 2])
    assert final_rows == [["H1", "H2", "H3", "H4"], ["a", "1", "b", "2"]]
    assert warnings == []


def test_stitch_landscape_refuses_to_merge_on_unresolvable_row_count_mismatch():
    left = [["H1"], ["a"], ["b"]]
    right = [["H2"], ["c"]]  # missing a row, and no empty-row candidate to explain it
    final_rows, warnings = stitch_landscape([left, right], expected_col_counts=[1, 1])
    assert final_rows == []
    assert any("disagree on row count" in w for w in warnings)


def test_stitch_landscape_recovers_one_unambiguous_empty_row():
    # left/right must actually DISAGREE on row count for reconciliation to trigger at all —
    # right has one genuine extra (fully empty) row, left doesn't.
    left = [["H1"], ["a"], ["b"]]
    right = [["H2"], ["c"], [""], ["d"]]
    final_rows, warnings = stitch_landscape([left, right], expected_col_counts=[1, 1])
    assert final_rows == [["H1", "H2"], ["a", "c"], ["b", "d"]]
    assert any("dropped 1 fully-empty row" in w for w in warnings)


def test_render_html_produces_a_table_with_matching_row_lengths():
    html = render_html([["H1", "H2"], ["a", "1"]])
    assert html.startswith("<table>")
    assert html.count("<tr>") == 2
    assert "<th>H1</th>" in html and "<th>H2</th>" in html
    assert "<td>a</td>" in html and "<td>1</td>" in html


def test_detect_label_chunk_index_finds_the_text_heavy_chunk_regardless_of_position():
    numeric_chunk = [["H1", "H2"], ["-", "1,234"], ["-", "5,678"]]
    label_chunk = [["H3", "H4"], ["Row label one", "note"], ["Row label two", "note"]]
    # label chunk passed SECOND — detection must not assume it's always first/left.
    assert detect_label_chunk_index([numeric_chunk, label_chunk]) == 1
    assert detect_label_chunk_index([label_chunk, numeric_chunk]) == 0


def test_detect_label_chunk_index_returns_none_when_no_chunk_has_real_text():
    numeric_a = [["H1"], ["-"], ["1,234"]]
    numeric_b = [["H2"], ["5,678"], ["-"]]
    assert detect_label_chunk_index([numeric_a, numeric_b]) is None
