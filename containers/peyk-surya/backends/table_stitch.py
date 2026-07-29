"""Q4's stitching logic (docs-personal/surya/improvement.md): merges N peyk-surya
predict_full outputs (one per split chunk) back into one table. Parses each chunk's returned
HTML into a grid first — never a plain text concatenation — and validates dimensions against
what the split was supposed to produce before merging: a mismatch means that chunk's
recognition drifted, flagged rather than merged positionally (wrong data is worse than missing
data for a financial table).

Emits HTML (render_html), not markdown — matches predict_full's own {"crop", "model", "html"}
contract so peyk-orchestrator's assemble_document() (normalize_digits(markdownify(full_html),
lang)) handles it unchanged.

detect_label_chunk_index replaces an earlier hardcoded "always reverse to put the pixel-right
chunk first" RTL assumption (wrong for an LTR table). Row labels are always structurally first
in reading order regardless of script direction — leftmost for LTR, rightmost-but-still-first
for RTL — so detecting which chunk's content is mostly non-numeric text, rather than assuming
a physical side, generalizes to both without knowing the document's script direction at all.

Ported from z_surya_exploration/stitch.py."""
import re
from html.parser import HTMLParser

_NUMERIC_LIKE_RE = re.compile(r"^[\s\d,.\-\(\)٠-٩۰-۹]*$")


class _TableGridParser(HTMLParser):
    """Parses predict_full's <table><tr><td>/<th> HTML into rows of cell text. Ignores
    colspan/rowspan and strips inner tags (<b>, <span>) down to their text content."""

    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._current_row = []
        elif tag in ("td", "th"):
            self._current_cell = []

    def handle_endtag(self, tag):
        if tag == "tr" and self._current_row is not None:
            self.rows.append(self._current_row)
            self._current_row = None
        elif tag in ("td", "th") and self._current_cell is not None:
            text = "".join(self._current_cell).strip()
            if self._current_row is not None:
                self._current_row.append(text)
            self._current_cell = None

    def handle_data(self, data):
        if self._current_cell is not None:
            self._current_cell.append(data)


def parse_html_table(html: str) -> list[list[str]]:
    parser = _TableGridParser()
    parser.feed(html)
    return parser.rows


def _is_empty_row(row: list[str]) -> bool:
    """Strict: every cell must be truly empty (no text at all) — NOT "-", which is legitimate
    financial content (means "no entry"/zero). Only a genuinely empty row is safe to treat as
    the known blank-separator/header-wrap artifact."""
    return all(cell == "" for cell in row)


def _trim_one_surplus_row(grid: list[list[str]], expected_count: int) -> tuple[list[list[str]], str | None]:
    """If grid has exactly one more row than expected_count and exactly one of its rows is
    fully empty, drop that row."""
    surplus = len(grid) - expected_count
    if surplus <= 0:
        return grid, None
    empty_indices = [idx for idx, row in enumerate(grid) if _is_empty_row(row)]
    if surplus == 1 and len(empty_indices) == 1:
        idx = empty_indices[0]
        return grid[:idx] + grid[idx + 1 :], f"dropped 1 fully-empty row (index {idx}) to reconcile row count"
    return grid, f"{surplus} extra row(s) but {len(empty_indices)} fully-empty candidate(s) — ambiguous, not auto-reconciling"


def stitch_portrait(chunk_grids: list[list[list[str]]], expected_row_counts: list[int]) -> tuple[list[list[str]], list[str]]:
    warnings = []
    if not chunk_grids or not chunk_grids[0]:
        return [], ["chunk 0 is empty — cannot establish header"]

    header = chunk_grids[0][0]
    ncols = len(header)
    final_rows = [header]

    for i, (grid, expected_count) in enumerate(zip(chunk_grids, expected_row_counts)):
        body = grid[1:] if grid else []
        if len(body) != expected_count:
            trimmed, note = _trim_one_surplus_row(body, expected_count)
            if note:
                warnings.append(f"chunk {i}: {note}")
            body = trimmed
            if len(body) != expected_count:
                warnings.append(f"chunk {i}: expected {expected_count} body rows, got {len(body)}")
        for r_idx, row in enumerate(body):
            if len(row) != ncols:
                warnings.append(f"chunk {i} row {r_idx}: expected {ncols} cols, got {len(row)} ({row})")
        final_rows.extend(body)

    return final_rows, warnings


def _reconcile_row_counts(chunk_grids: list[list[list[str]]]) -> tuple[list[list[list[str]]], list[str]]:
    """Chunks with MORE rows than the minimum are assumed to have a spurious extra row — never
    the other way around, since a chunk with fewer rows means real data went missing. Only
    trims when there's exactly ONE empty-row candidate per surplus row (unambiguous)."""
    warnings = []
    target = min(len(g) for g in chunk_grids)
    fixed = []
    for i, grid in enumerate(chunk_grids):
        trimmed, note = _trim_one_surplus_row(grid, target)
        if note:
            warnings.append(f"chunk {i}: {note}")
        fixed.append(trimmed)
    return fixed, warnings


def stitch_landscape(chunk_grids: list[list[list[str]]], expected_col_counts: list[int]) -> tuple[list[list[str]], list[str]]:
    """chunk_grids must already be in final reading order (see detect_label_chunk_index for
    how the caller decides that order) — this function just joins in the order given. No
    header special-casing — the header row is row 0 of every chunk's own grid, since each
    chunk naturally contains its own column-slice of it."""
    warnings = []
    if not chunk_grids or any(not g for g in chunk_grids):
        return [], ["one or more chunks returned an empty grid"]

    row_counts = [len(g) for g in chunk_grids]
    if len(set(row_counts)) != 1:
        chunk_grids, reconcile_warnings = _reconcile_row_counts(chunk_grids)
        warnings.extend(reconcile_warnings)
        row_counts = [len(g) for g in chunk_grids]
        if len(set(row_counts)) != 1:
            warnings.append(f"chunks still disagree on row count {row_counts} after reconciliation attempt — alignment by row index is unsafe, not merging")
            return [], warnings

    # Column-ragged chunks are never merged positionally: one short row shifts every later
    # chunk's values under the wrong header labels for that row, silently. Same bail-out as the
    # row-count mismatch above — the caller falls back to unstitched chunks + a continuation
    # marker. No padding, no guessing (wrong data is worse than missing data here).
    ragged = False
    for i, (grid, expected_count) in enumerate(zip(chunk_grids, expected_col_counts)):
        for r_idx, row in enumerate(grid):
            if len(row) != expected_count:
                warnings.append(f"chunk {i} row {r_idx}: expected {expected_count} cols, got {len(row)} ({row})")
                ragged = True
    if ragged:
        warnings.append("column counts are ragged — positional merge would misalign values under the wrong headers, not merging")
        return [], warnings

    final_rows = [sum((grid[r_idx] for grid in chunk_grids), []) for r_idx in range(row_counts[0])]
    return final_rows, warnings


def _is_numeric_like(cell: str) -> bool:
    return bool(_NUMERIC_LIKE_RE.fullmatch(cell.strip()))


def _column_textiness(grid: list[list[str]], col_idx: int) -> float:
    body = grid[1:]  # skip header
    if not body:
        return 0.0
    text_count = sum(1 for row in body if col_idx < len(row) and row[col_idx] and not _is_numeric_like(row[col_idx]))
    return text_count / len(body)


def detect_label_chunk_index(chunk_grids: list[list[list[str]]]) -> int | None:
    """Finds which chunk holds the row-label column, from content alone — no script-direction
    assumption. Row labels are always structurally first in reading order in any convention
    (leftmost for LTR, rightmost-but-first for RTL), so "which chunk has a mostly-non-numeric
    column" identifies the chunk that should go first when merging, regardless of which
    physical side of the crop it came from. Returns None if no chunk has a clearly text-heavy
    column (e.g. every chunk is purely numeric — not expected for the validated num_splits=2
    default, where exactly one chunk should contain the true edge/label column)."""
    best_chunk, best_score = None, 0.0
    for i, grid in enumerate(chunk_grids):
        if not grid or not grid[0]:
            continue
        ncols = len(grid[0])
        chunk_best = max((_column_textiness(grid, c) for c in range(ncols)), default=0.0)
        if chunk_best > best_score:
            best_score, best_chunk = chunk_best, i
    return best_chunk


def render_html(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    header = rows[0]
    lines = ["<table>", "<tbody>", "<tr>"]
    lines += [f"<th>{cell}</th>" for cell in header]
    lines.append("</tr>")
    for row in rows[1:]:
        lines.append("<tr>")
        lines += [f"<td>{cell}</td>" for cell in row]
        lines.append("</tr>")
    lines += ["</tbody>", "</table>"]
    return "\n".join(lines)
