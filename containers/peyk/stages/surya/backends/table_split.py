"""Q2's split-boundary logic (docs-personal/surya/improvement.md): given peyk-tsr's row/column
band detection for a table crop, find where to cut it — at the boundary nearest the geometric
center (the documented default, num_splits=2), or at N equally-spaced boundaries for
experimentation (num_splits>2, not the default). Every boundary-picking path is merged-cell
aware: a cut is never placed where a cell's row_span/col_span would be sliced in half (see
_boundary_is_safe) — peyk-tsr's regularized_cells() carries row_span/col_span per cell in its
_aug.json output for exactly this purpose.

Operates entirely on plain dicts matching peyk-tsr's _aug.json shape — rows/cols entries are
{"row"|"col": int, "bbox": [x0,y0,x1,y1]}; cells entries additionally have "row_span"/
"col_span". No file I/O here — split_rows/split_cols take and return in-memory PIL images,
since this runs in-process inside run.py's --stage table-full handling, not as its own CLI.

Ported from z_surya_exploration/split_by_tsr.py (see docs-personal/surya/improvement.md Q2)."""
from PIL import Image


def _span_indices(axis_key: str) -> tuple[int, int]:
    return (1, 3) if axis_key == "row" else (0, 2)  # bbox indices for this axis's span


def body_center(bands: list[dict], axis_key: str) -> float:
    """Geometric center of the bands' own extent (first band's top/left to last band's
    bottom/right) — the correct `center` to pass to center_split_index for row bands, where
    `bands` already excludes the header band. For column bands (no header exclusion), callers
    can just use image_width/2 directly instead."""
    lo, hi = _span_indices(axis_key)
    return (bands[0]["bbox"][lo] + bands[-1]["bbox"][hi]) / 2


def _boundary_is_safe(bands: list[dict], k: int, cells: list[dict] | None, axis_key: str) -> bool:
    """True if cutting between bands[k-1] and bands[k] doesn't slice through any cell whose
    row_span/col_span covers both sides of that boundary. cells=None (no span data available)
    is treated as always-safe rather than blocking every split."""
    if not cells:
        return True
    span_key = f"{axis_key}_span"
    before_idx = bands[k - 1][axis_key]
    after_idx = bands[k][axis_key]
    for cell in cells:
        cell_start = cell[axis_key]
        cell_end = cell_start + cell.get(span_key, 1) - 1
        if cell_start <= before_idx and cell_end >= after_idx:
            return False
    return True


def center_split_index(bands: list[dict], center: float, axis_key: str, cells: list[dict] | None = None) -> int:
    """Returns the split index k (1 <= k <= len(bands)-1) such that the cut between
    bands[k-1] and bands[k] falls nearest `center`. Candidates that would slice through a
    merged cell are skipped entirely, not just deprioritized. Raises ValueError if every
    candidate boundary is unsafe."""
    lo, hi = _span_indices(axis_key)
    best_k, best_dist = None, None
    for k in range(1, len(bands)):
        if not _boundary_is_safe(bands, k, cells, axis_key):
            continue
        boundary = (bands[k - 1]["bbox"][hi] + bands[k]["bbox"][lo]) / 2
        dist = abs(boundary - center)
        if best_dist is None or dist < best_dist:
            best_dist, best_k = dist, k
    if best_k is None:
        raise ValueError(f"no safe split boundary for axis={axis_key} — every candidate cut straddles a merged cell spanning the whole table")
    return best_k


def nearest_safe_boundary(bands: list[dict], candidate_k: int, cells: list[dict] | None, axis_key: str) -> int:
    """For boundaries not chosen by center_split_index (max_per_chunk_to_boundaries'
    equally-spaced targets) — nudges a candidate that would slice through a merged cell to the
    nearest index (either direction) that doesn't."""
    n = len(bands)
    if _boundary_is_safe(bands, candidate_k, cells, axis_key):
        return candidate_k
    for delta in range(1, n):
        for k in (candidate_k - delta, candidate_k + delta):
            if 1 <= k <= n - 1 and _boundary_is_safe(bands, k, cells, axis_key):
                return k
    raise ValueError(f"no safe split boundary for axis={axis_key} — every candidate cut straddles a merged cell spanning the whole table")


def max_per_chunk_to_boundaries(
    n: int, max_per_chunk: int, bands: list[dict] | None = None, cells: list[dict] | None = None, axis_key: str | None = None
) -> list[int]:
    """bands/cells/axis_key are optional — when given, each equally-spaced boundary is nudged
    off a merged cell it would otherwise straddle. Omit them for the plain, span-unaware
    behavior."""
    raw = list(range(max_per_chunk, n, max_per_chunk))
    if bands is None:
        return raw
    seen: set[int] = set()
    safe = []
    for k in raw:
        k_safe = nearest_safe_boundary(bands, k, cells, axis_key)
        if k_safe not in seen:
            seen.add(k_safe)
            safe.append(k_safe)
    return sorted(safe)


def compute_chunks(bands: list[dict], boundaries: list[int], axis_key: str) -> list[tuple[int, int | None, list[int]]]:
    """bands: sorted list of body bands (header already excluded for rows). boundaries: sorted
    split indices into bands. Returns (span0, span1, indices) per chunk — span1=None for the
    last chunk (crop to image edge). Cut lines are the midpoint between consecutive bands, so
    they always land in whitespace, never inside a band."""
    lo, hi = _span_indices(axis_key)
    cuts = [0, *boundaries, len(bands)]
    groups = [bands[cuts[i] : cuts[i + 1]] for i in range(len(cuts) - 1)]

    chunks = []
    for gi, group in enumerate(groups):
        indices = [b[axis_key] for b in group]
        if gi == 0:
            span0 = 0
        else:
            prev_last = groups[gi - 1][-1]["bbox"][hi]
            this_first = group[0]["bbox"][lo]
            span0 = int((prev_last + this_first) / 2)

        if gi == len(groups) - 1:
            span1 = None
        else:
            this_last = group[-1]["bbox"][hi]
            next_first = groups[gi + 1][0]["bbox"][lo]
            span1 = int((this_last + next_first) / 2)

        chunks.append((span0, span1, indices))
    return chunks


def split_rows(image: Image.Image, rows: list[dict], boundaries: list[int]) -> list[tuple[Image.Image, list[int]]]:
    """rows[0] is the header band; body rows are rows[1:]. Header band is cropped once (y=0 to
    the midpoint between row 0 and row 1) and pasted onto every chunk. Returns
    [(chunk_image, row_indices), ...] in chunk order."""
    w, h = image.size
    header_bottom = int((rows[0]["bbox"][3] + rows[1]["bbox"][1]) / 2)
    header_crop = image.crop((0, 0, w, header_bottom))

    body = rows[1:]
    chunks = compute_chunks(body, boundaries, axis_key="row")

    result = []
    for gi, (y0, y1, row_indices) in enumerate(chunks):
        # compute_chunks returns span0=0 (absolute image top) for the first chunk — correct
        # for split_cols (no header, chunk 0 legitimately starts at the image edge), but wrong
        # here: y=0 is the header band's own region, already pasted separately above. Without
        # this override, chunk 0's body_crop restarts at the image top and the header band
        # ends up rendered twice (once as header_crop, once embedded in body_crop) — confirmed
        # via a real crop (bdc_sample_true_scanned__r18), not caught by earlier synthetic tests.
        crop_y0 = header_bottom if gi == 0 else y0
        body_crop = image.crop((0, crop_y0, w, y1 if y1 is not None else h))
        combined = Image.new("RGB", (w, header_crop.height + body_crop.height), "white")
        combined.paste(header_crop, (0, 0))
        combined.paste(body_crop, (0, header_crop.height))
        result.append((combined, row_indices))
    return result


def split_cols(image: Image.Image, cols: list[dict], boundaries: list[int]) -> list[tuple[Image.Image, list[int]]]:
    """No header reattachment — the header row is row 0 of the table and gets column-sliced
    along with every other row, so each chunk already contains its own slice of it. Returns
    [(chunk_image, col_indices), ...] in LEFT-TO-RIGHT pixel order (chunk 0 = leftmost) —
    caller (Task 7) is responsible for reordering to reading order before stitching."""
    w, h = image.size
    chunks = compute_chunks(cols, boundaries, axis_key="col")

    result = []
    for x0, x1, col_indices in chunks:
        crop = image.crop((x0, 0, x1 if x1 is not None else w, h))
        result.append((crop, col_indices))
    return result
