# peyk-surya Smart Table-Full Recognition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `peyk-surya`'s `--stage table-full` self-correcting on blurry table crops — measure sharpness, and below threshold, split the crop, upscale each piece, recognize each independently, and stitch the results back into one HTML table, all inside `peyk-surya` itself, transparent to `peyk-orchestrator`.

**Architecture:** `peyk-surya` gains a new capability it doesn't have today — calling `peyk-tsr` itself via `docker run` (needs the host docker socket mounted into its own container, same access `peyk-orchestrator` currently has exclusively) — to get TableFormer's row/column structure for a crop it has independently judged too blurry to recognize in one shot. The algorithm (measure → split → upscale → dispatch → stitch) is ported near-verbatim from the validated `z_surya_exploration/process_table.py` prototype, adapted to run in-process inside `run.py`'s existing `--stage table-full` handling instead of shelling out per-step, and adapted to emit HTML (not markdown) so it drops into `peyk-orchestrator`'s existing `assemble_document()` unchanged. Chunk ordering for split tables is decided from the recognized content itself (which chunk holds text-heavy row labels) rather than an assumed script direction.

**Tech Stack:** Python 3.10, PIL/Pillow, numpy, `pytest`, Docker (docker-in-docker via mounted host socket), Surya-OCR-2 (`peyk-vllm-surya`), TableFormer (`peyk-tsr`).

## Global Constraints

- `sharpness_threshold_laplacian_var = 450` (exact value, from `docs-personal/surya/improvement.md` Q1 — do not re-derive or change it).
- `max_pixels = 6_291_456`, `pixel_safety_margin = 0.97`, `max_upscale_cap = 2.0`, `min_scale = 1.0` (exact values, from `docs-personal/surya/improvement.md` Q3 — do not change).
- Row-span/col-span guard on split boundaries is mandatory (see `docs-personal/surya/improvement.md`'s resolved row-span gap) — a chunk boundary must never straddle a cell whose `row_span`/`col_span` covers both sides of it.
- Stitching must emit an HTML `<table>` string matching `predict_full`'s existing `{"crop", "model", "html"}` output contract — never markdown. `peyk-orchestrator`'s `assemble_document()` runs the returned `html` through `markdownify`/`normalize_digits` itself; do not pre-convert.
- If a split table's chunks cannot be confidently stitched (dimension mismatch that reconciliation can't resolve), never fail the request and never merge positionally — return the unstitched chunk HTML tables concatenated with a plain-text marker line between them (`"يتبع الجدول التالي من الجدول السابق"` — "the following table is a continuation of the table above") rather than guessing.
- Only `--stage table-full` gains this new behavior. `--stage layout`/`tsr`/`ocr` and `--mode fullpage` are unchanged.
- `peyk-surya`'s existing output contract for `--stage table-full` (`{"crop": <name>, "model": "surya", "html": <str>}`, one JSON file per input crop, written to `--output`) does not change shape — only how `html` gets produced internally.
- Never modify `containers/peyk-tsr` or `containers/peyk-vllm-surya` in this plan — this integrates against their existing, already-working contracts (`peyk-tsr --model tableformer --input <dir> --output <dir>` producing `<stem>.json`/`<stem>_aug.json`; `peyk-vllm-surya` already running and reachable at `http://peyk-vllm-surya:8000/v1`).
- Every new pure-logic module (`sharpness.py`, `table_split.py`, `table_stitch.py`) must have real `pytest` unit tests with no mocks of the logic under test — only Docker/network calls are out of scope for unit tests (those get integration verification in the wiring tasks instead).

---

### Task 1: Sharpness measurement (`backends/sharpness.py`)

**Files:**
- Create: `containers/peyk-surya/backends/sharpness.py`
- Create: `containers/peyk-surya/tests/test_sharpness.py`
- Modify: `containers/peyk-surya/requirements.txt` (add `numpy`)

**Interfaces:**
- Produces: `laplacian_variance(image: PIL.Image.Image) -> float` — used by Task 7's dispatch logic to decide whether a crop needs correction.

- [ ] **Step 1: Add numpy to requirements**

Append to `containers/peyk-surya/requirements.txt`:

```
# Used by backends/sharpness.py's laplacian_variance (Q1's blur-detection trigger, see
# docs-personal/surya/improvement.md) — no other dependency here already pulls it in.
numpy>=1.26,<2.0
```

- [ ] **Step 2: Write the failing test**

Create `containers/peyk-surya/tests/test_sharpness.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image, ImageFilter

from backends.sharpness import laplacian_variance


def _checkerboard(size=200, square=10):
    """A sharp, high-frequency image: alternating black/white squares. Should score high."""
    image = Image.new("L", (size, size), 255)
    pixels = image.load()
    for y in range(size):
        for x in range(size):
            if (x // square + y // square) % 2 == 0:
                pixels[x, y] = 0
    return image.convert("RGB")


def test_sharp_image_scores_high():
    sharp = _checkerboard()
    assert laplacian_variance(sharp) > 1000


def test_blurring_reduces_the_score():
    sharp = _checkerboard()
    blurred = sharp.filter(ImageFilter.GaussianBlur(radius=3))
    assert laplacian_variance(blurred) < laplacian_variance(sharp) / 10


def test_flat_image_scores_near_zero():
    flat = Image.new("RGB", (200, 200), (255, 255, 255))
    assert laplacian_variance(flat) < 1.0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd containers/peyk-surya && python3 -m pytest tests/test_sharpness.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backends.sharpness'`

- [ ] **Step 4: Write the implementation**

Create `containers/peyk-surya/backends/sharpness.py`:

```python
"""Q1's blur-detection trigger (docs-personal/surya/improvement.md): variance of a 3x3
Laplacian-filtered image. Chosen over three other sharpness metrics tested (tenengrad,
brenner, modified_laplacian) for giving the widest relative separation between confirmed
fail/pass cases (3.39x vs. 2.60x/1.69x/1.54x) — see that doc's Q1 for the full evidence and
conceptual justification. Ported from z_surya_exploration/measure_noise.py — only this one
metric, the other three were comparison-only during metric selection, never part of the
actual decision logic."""
import numpy as np
from PIL import Image, ImageFilter

LAPLACIAN_KERNEL = ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1)


def laplacian_variance(image: Image.Image) -> float:
    filtered = image.convert("L").filter(LAPLACIAN_KERNEL)
    return float(np.asarray(filtered, dtype=np.float64).var())
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd containers/peyk-surya && python3 -m pytest tests/test_sharpness.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add containers/peyk-surya/backends/sharpness.py containers/peyk-surya/tests/test_sharpness.py containers/peyk-surya/requirements.txt
git commit -m "feat(peyk-surya): add laplacian_variance sharpness metric (Q1 trigger)"
```

---

### Task 2: Split-boundary logic (`backends/table_split.py`)

**Files:**
- Create: `containers/peyk-surya/backends/table_split.py`
- Create: `containers/peyk-surya/tests/test_table_split.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `body_center(bands: list[dict], axis_key: str) -> float`
  - `center_split_index(bands: list[dict], center: float, axis_key: str, cells: list[dict] | None = None) -> int`
  - `max_per_chunk_to_boundaries(n: int, max_per_chunk: int, bands: list[dict] | None = None, cells: list[dict] | None = None, axis_key: str | None = None) -> list[int]`
  - `compute_chunks(bands: list[dict], boundaries: list[int], axis_key: str) -> list[tuple[int, int | None, list[int]]]`
  - `split_rows(image: PIL.Image.Image, rows: list[dict], boundaries: list[int], ) -> list[tuple[PIL.Image.Image, list[int]]]`
  - `split_cols(image: PIL.Image.Image, cols: list[dict], boundaries: list[int]) -> list[tuple[PIL.Image.Image, list[int]]]`

  All consumed by Task 7. `bands`/`cells` are plain dicts matching `peyk-tsr`'s `_aug.json`
  shape: rows/cols entries are `{"row"|"col": int, "bbox": [x0,y0,x1,y1]}`; cells entries are
  `{"row": int, "col": int, "row_span": int, "col_span": int, "bbox": [...]}`.

  **Note the signature difference from the `z_surya_exploration` prototype**: `split_rows`/
  `split_cols` here take and return in-memory `PIL.Image.Image` objects (no `out_dir`/
  `prefix`/file-writing, no `print()`) — this module is called in-process by `run.py`, not run
  as its own CLI script.

- [ ] **Step 1: Write the failing tests**

Create `containers/peyk-surya/tests/test_table_split.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd containers/peyk-surya && python3 -m pytest tests/test_table_split.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backends.table_split'`

- [ ] **Step 3: Write the implementation**

Create `containers/peyk-surya/backends/table_split.py`:

```python
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
    for y0, y1, row_indices in chunks:
        body_crop = image.crop((0, y0, w, y1 if y1 is not None else h))
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd containers/peyk-surya && python3 -m pytest tests/test_table_split.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add containers/peyk-surya/backends/table_split.py containers/peyk-surya/tests/test_table_split.py
git commit -m "feat(peyk-surya): add merged-cell-aware split-boundary logic (Q2)"
```

---

### Task 3: Upscale logic (`backends/table_scale.py`)

**Files:**
- Create: `containers/peyk-surya/backends/table_scale.py`
- Create: `containers/peyk-surya/tests/test_table_scale.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `safe_upscale(image: PIL.Image.Image, max_upscale_cap: float = 2.0) -> PIL.Image.Image` — consumed by Task 7. Returns the same image object unchanged (not a copy) when no upscale is needed, so callers can check `result is image` if they need to know whether scaling happened.

- [ ] **Step 1: Write the failing tests**

Create `containers/peyk-surya/tests/test_table_scale.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image

from backends.table_scale import MAX_PIXELS, MIN_SCALE, PIXEL_SAFETY_MARGIN, safe_upscale


def test_small_image_gets_upscaled_toward_the_cap():
    image = Image.new("RGB", (500, 300), "white")  # 150,000px, tiny relative to MAX_PIXELS
    result = safe_upscale(image, max_upscale_cap=2.0)
    assert result.size == (1000, 600)  # exactly 2.0x, the cap — plenty of pixel headroom below it


def test_upscale_never_exceeds_the_cap_even_with_huge_headroom():
    image = Image.new("RGB", (10, 10), "white")
    result = safe_upscale(image, max_upscale_cap=1.5)
    assert result.size == (15, 15)


def test_image_already_near_the_pixel_ceiling_is_left_unchanged():
    # sqrt(MAX_PIXELS * PIXEL_SAFETY_MARGIN) ~= 2470px per side for a square image -> scale_cap ~= 1.0
    side = int((MAX_PIXELS * PIXEL_SAFETY_MARGIN) ** 0.5)
    image = Image.new("RGB", (side, side), "white")
    result = safe_upscale(image)
    assert result is image  # MIN_SCALE floor: never downscale, and no meaningful upscale headroom either


def test_never_downscales_a_crop_already_over_the_pixel_ceiling():
    # Deliberately oversized crop (scale_cap < 1.0) — must still never shrink it.
    side = int((MAX_PIXELS * PIXEL_SAFETY_MARGIN) ** 0.5) + 500
    image = Image.new("RGB", (side, side), "white")
    result = safe_upscale(image)
    assert result is image
    assert result.size == (side, side)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd containers/peyk-surya && python3 -m pytest tests/test_table_scale.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backends.table_scale'`

- [ ] **Step 3: Write the implementation**

Create `containers/peyk-surya/backends/table_scale.py`:

```python
"""Q3's upscale rule (docs-personal/surya/improvement.md): the largest safe upscale under
peyk-vllm-surya's max_pixels cap, never downscaling. max_upscale_cap=2.0 is a
confirmed-necessary defensive ceiling, not arbitrary caution — removing it reproduces a hard
"Inference error: 'NoneType' object has no attribute 'chat'" failure when a chunk's actual
pixel count (after independently rounding width/height up) creeps past the real cap. Ported
from z_surya_exploration/process_table.py's safe_upscale."""
import math

from PIL import Image

MAX_PIXELS = 6_291_456  # peyk-vllm-surya's --mm-processor-kwargs max_pixels (start.sh) — do not change without re-verifying against that config
PIXEL_SAFETY_MARGIN = 0.97
MAX_UPSCALE_CAP = 2.0
MIN_SCALE = 1.0


def safe_upscale(image: Image.Image, max_upscale_cap: float = MAX_UPSCALE_CAP) -> Image.Image:
    w, h = image.size
    scale_cap = math.sqrt(MAX_PIXELS * PIXEL_SAFETY_MARGIN / (w * h))
    scale = min(scale_cap, max_upscale_cap)
    if scale <= MIN_SCALE:
        return image

    new_size = (round(w * scale), round(h * scale))
    return image.resize(new_size, Image.LANCZOS)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd containers/peyk-surya && python3 -m pytest tests/test_table_scale.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add containers/peyk-surya/backends/table_scale.py containers/peyk-surya/tests/test_table_scale.py
git commit -m "feat(peyk-surya): add Q3 safe-upscale logic"
```

---

### Task 4: Stitching logic, HTML output (`backends/table_stitch.py`)

**Files:**
- Create: `containers/peyk-surya/backends/table_stitch.py`
- Create: `containers/peyk-surya/tests/test_table_stitch.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure logic over parsed grids/HTML strings).
- Produces:
  - `parse_html_table(html: str) -> list[list[str]]`
  - `stitch_portrait(chunk_grids: list[list[list[str]]], expected_row_counts: list[int]) -> tuple[list[list[str]], list[str]]`
  - `stitch_landscape(chunk_grids: list[list[list[str]]], expected_col_counts: list[int]) -> tuple[list[list[str]], list[str]]`
  - `render_html(rows: list[list[str]]) -> str` — **replaces the prototype's `render_markdown`**; emits an HTML `<table>` string matching `predict_full`'s own output shape.
  - `detect_label_chunk_index(chunk_grids: list[list[list[str]]]) -> int | None` — **new**, replaces the prototype's hardcoded RTL chunk-order reversal (see Task 7 for how it's used).

  All consumed by Task 7.

- [ ] **Step 1: Write the failing tests**

Create `containers/peyk-surya/tests/test_table_stitch.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd containers/peyk-surya && python3 -m pytest tests/test_table_stitch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backends.table_stitch'`

- [ ] **Step 3: Write the implementation**

Create `containers/peyk-surya/backends/table_stitch.py`:

```python
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

    for i, (grid, expected_count) in enumerate(zip(chunk_grids, expected_col_counts)):
        for r_idx, row in enumerate(grid):
            if len(row) != expected_count:
                warnings.append(f"chunk {i} row {r_idx}: expected {expected_count} cols, got {len(row)} ({row})")

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd containers/peyk-surya && python3 -m pytest tests/test_table_stitch.py -v`
Expected: PASS (10 passed)

- [ ] **Step 5: Commit**

```bash
git add containers/peyk-surya/backends/table_stitch.py containers/peyk-surya/tests/test_table_stitch.py
git commit -m "feat(peyk-surya): add HTML-emitting stitch logic with content-based chunk ordering"
```

---

### Task 5: Give `peyk-surya` docker-in-docker access to call `peyk-tsr`

**Files:**
- Modify: `containers/peyk-surya/Dockerfile`
- Modify: `containers/peyk-orchestrator/stages.py`
- Modify: `containers/peyk-orchestrator/pipeline.py:377-389` (`dispatch_table_full_batch`)

**Interfaces:**
- Produces: `peyk-surya`'s container has the `docker` CLI binary installed, and when
  `peyk-orchestrator` dispatches it for the `table-full` role, it's launched with the host
  docker socket mounted — both required before Task 6 can call `docker run peyk-tsr:dev` from
  inside `peyk-surya`'s own process.

- [ ] **Step 1: Install the docker CLI in peyk-surya's image**

In `containers/peyk-surya/Dockerfile`, after the `FROM python:3.10-slim` line and before
`WORKDIR /app`, add:

```dockerfile
# Installs only the docker CLI binary (not the daemon) — peyk-surya needs this to dispatch
# peyk-tsr itself (docker-in-docker via the host socket mounted at runtime, see
# containers/peyk-orchestrator/pipeline.py's dispatch_table_full_batch and stages.py) when a
# table crop is too blurry to recognize in one shot and needs TableFormer's row/column
# structure to know where to split it. See docs-personal/surya/improvement.md's "Before
# integrating" section, decision 1.
RUN apt-get update && apt-get install -y --no-install-recommends docker.io && rm -rf /var/lib/apt/lists/*
```

- [ ] **Step 2: Verify the image builds with the docker CLI present**

Run: `cd containers/peyk-surya && docker build -t peyk-surya:dev . && docker run --rm --entrypoint docker peyk-surya:dev --version`
Expected: prints a `Docker version ...` line (confirms the CLI binary is present and runnable inside the image) — the entrypoint override is only for this one verification command, `ENTRYPOINT ["python3", "run.py"]` is unchanged for real invocations.

- [ ] **Step 3: Mount the host docker socket for peyk-surya's table-full dispatch**

In `containers/peyk-orchestrator/pipeline.py`, find `dispatch_table_full_batch` (around line
377):

```python
    backend = config.tsr.backend
    if backend == "surya":
        extra_args, extra_docker_args = ["--stage", "table-full"], None
    else:
        extra_args, extra_docker_args = ["--role", "table"], _vlm_credential_docker_args(backend)
```

Replace the `backend == "surya"` branch's `extra_docker_args` value:

```python
    backend = config.tsr.backend
    if backend == "surya":
        # peyk-surya's --stage table-full now decides internally (per crop) whether a table
        # needs splitting, and if so calls peyk-tsr itself for TableFormer's row/column
        # structure — needs the host docker socket to do that docker-in-docker dispatch. See
        # docs-personal/surya/improvement.md's "Before integrating" section, decision 1.
        extra_args = ["--stage", "table-full"]
        extra_docker_args = ["-v", "/var/run/docker.sock:/var/run/docker.sock"]
    else:
        extra_args, extra_docker_args = ["--role", "table"], _vlm_credential_docker_args(backend)
```

- [ ] **Step 4: Confirm stages.py's run_docker_stage needs no change**

Read `containers/peyk-orchestrator/stages.py`'s `run_docker_stage` function (around line 55) —
confirm the existing `if extra_docker_args: cmd += extra_docker_args` block (around line 106)
already splices any list of extra docker flags in before the image name, with no
`peyk-surya`-specific logic needed. No edit required here; this step is a verification read,
not a code change.

- [ ] **Step 5: Commit**

```bash
git add containers/peyk-surya/Dockerfile containers/peyk-orchestrator/pipeline.py
git commit -m "feat: mount docker socket into peyk-surya for its table-full dispatch"
```

---

### Task 6: `peyk-surya` calls `peyk-tsr` itself (`backends/tsr_dispatch.py`)

**Files:**
- Create: `containers/peyk-surya/backends/tsr_dispatch.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `dispatch_tsr(image_path: Path, workdir: Path) -> dict` — given a single crop file
  path and a scratch directory that is a real subpath of the shared workdir volume (so a
  sibling `peyk-tsr` container launched via `--volumes-from` sees the same files at the same
  paths — see `containers/peyk-orchestrator/stages.py`'s own comment on why `--volumes-from`
  is used instead of a bind-mount path translation), returns the parsed `_aug.json` dict
  (`{"rows": [...], "cols": [...], "cells": [...]}`). Consumed by Task 7.

- [ ] **Step 1: Write the implementation**

Create `containers/peyk-surya/backends/tsr_dispatch.py`:

```python
"""peyk-surya calling peyk-tsr itself — a new capability, not something this container could
do before (see docs-personal/surya/improvement.md's "Before integrating" section, decision 1).
Only used for --stage table-full, and only for a crop Q1's sharpness check has already flagged
as needing a split (see run.py) — a sharp crop never triggers this at all.

Requires: the host docker socket mounted into this container (see
containers/peyk-orchestrator/pipeline.py's dispatch_table_full_batch and stages.py) and the
docker CLI installed in this image (see Dockerfile). Dispatches peyk-tsr the same way
peyk-orchestrator's own stages.py dispatches every sibling stage container — --volumes-from
ORCHESTRATOR_CONTAINER_NAME (not a host-path bind mount; see that container's own comment for
why: a literal `-v <path>:/data/in` here would resolve against the HOST filesystem through the
shared docker socket, not this container's own filesystem view, silently binding an empty
directory if the paths don't already point at a real host-visible path).
"""
import json
import shutil
import subprocess
from pathlib import Path

# Must match peyk-orchestrator/stages.py's own constants exactly — this dispatch has to join
# the same network and reference the same well-known orchestrator container name that other
# sibling stage containers already do.
PEYK_NETWORK = "peyk-net"
ORCHESTRATOR_CONTAINER_NAME = "peyk-orchestrator-run"
TSR_IMAGE = "peyk-tsr:dev"


def dispatch_tsr(image_path: Path, workdir: Path) -> dict:
    """workdir must be a real subpath of the shared workdir volume peyk-surya itself was
    launched with --volumes-from ORCHESTRATOR_CONTAINER_NAME for (i.e. under the same
    directory tree as this container's own --input/--output arguments) — not an arbitrary
    tempfile.TemporaryDirectory() path, which would only exist inside peyk-surya's own
    filesystem and be invisible to the peyk-tsr sibling this function launches."""
    in_dir = workdir / "tsr_dispatch_in"
    out_dir = workdir / "tsr_dispatch_out"
    for d in (in_dir, out_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)

    shutil.copy(image_path, in_dir / image_path.name)

    subprocess.run(
        [
            "docker", "run", "--rm", "--gpus", "all",
            "--network", PEYK_NETWORK,
            "--volumes-from", ORCHESTRATOR_CONTAINER_NAME,
            TSR_IMAGE,
            "--model", "tableformer",
            "--input", str(in_dir),
            "--output", str(out_dir),
        ],
        check=True,
    )

    aug_path = out_dir / f"{image_path.stem}_aug.json"
    return json.loads(aug_path.read_text(encoding="utf-8"))
```

- [ ] **Step 2: Verify against a real crop, with peyk-vllm-surya/peyk-tsr already built**

This module has no pure-logic unit test — it's a thin docker-dispatch wrapper, verified
against the real containers instead, same as every other integration point in this project.

Run (from the repo root, with `peyk-tsr:dev` already built and the `peyk-orchestrator-run`
container name available — this check runs `dispatch_tsr` standalone against a throwaway
container mimicking that name for the purpose of this one verification):

```bash
docker rm -f peyk-orchestrator-run 2>/dev/null || true
mkdir -p /tmp/peyk_tsr_dispatch_test/workdir
docker run -d --name peyk-orchestrator-run -v /tmp/peyk_tsr_dispatch_test/workdir:/workdir alpine sleep 3600
docker network create peyk-net 2>/dev/null || true
docker network connect peyk-net peyk-orchestrator-run 2>/dev/null || true
python3 -c "
import sys
sys.path.insert(0, 'containers/peyk-surya')
from pathlib import Path
from backends.tsr_dispatch import dispatch_tsr
aug = dispatch_tsr(Path('hotstorage/workdir/table_full_in/cib_sample__r12.png'), Path('/tmp/peyk_tsr_dispatch_test/workdir'))
assert 'rows' in aug and 'cols' in aug and 'cells' in aug
assert 'row_span' in aug['cells'][0]
print('OK:', len(aug['rows']), 'rows,', len(aug['cols']), 'cols,', len(aug['cells']), 'cells')
"
docker rm -f peyk-orchestrator-run
```

Expected: `OK: <N> rows, <M> cols, <K> cells` with no exception — confirms `dispatch_tsr` can
launch `peyk-tsr` as a real sibling container and read back its `_aug.json` with `row_span`
present per cell.

- [ ] **Step 3: Commit**

```bash
git add containers/peyk-surya/backends/tsr_dispatch.py
git commit -m "feat(peyk-surya): add capability to dispatch peyk-tsr itself"
```

---

### Task 7: Wire the smart flow into `--stage table-full`

**Files:**
- Modify: `containers/peyk-surya/run.py`

**Interfaces:**
- Consumes: `laplacian_variance` (Task 1), `body_center`/`center_split_index`/
  `max_per_chunk_to_boundaries`/`split_rows`/`split_cols` (Task 2), `safe_upscale` (Task 3),
  `parse_html_table`/`stitch_portrait`/`stitch_landscape`/`render_html`/
  `detect_label_chunk_index` (Task 4), `dispatch_tsr` (Task 6).
- Produces: `_process_crop_table_full`'s output contract is unchanged
  (`{"crop": <name>, "model": "surya", "html": <str>}` written to `<output>/<stem>.json`) —
  only how `html` is produced changes.

- [ ] **Step 1: Read the current implementation to modify precisely**

`containers/peyk-surya/run.py`'s `_process_crop_table_full` and `run_stage_table_full`
(lines 306-346) currently always call `client.predict_table_full(image)` directly:

```python
def _html_from_table_full_result(result) -> str:
    """TableRecPredictor.predict_full's result shape is NOT confirmed the way
    RecognitionPredictor's PageOCRResult/BlockOCRResult shape now is (see
    _blocks_from_recognition_result's docstring for that investigation) — this guess
    (flat `.html`/`.text` attribute) hasn't been checked against Surya's real source or a live
    response. Expect to revisit this the same way _text_from_recognition_result's original
    guess had to be, once this stage actually runs against the live server."""
    return getattr(result, "html", None) or getattr(result, "text", "") or ""


def _process_crop_table_full(crop_path: Path, client: SuryaClient, output_dir: Path) -> None:
    from PIL import Image

    image = Image.open(crop_path)
    table_html_result = client.predict_table_full(image)
    html = _html_from_table_full_result(table_html_result)

    out_path = output_dir / f"{crop_path.stem}.json"
    out_path.write_text(json.dumps({"crop": crop_path.name, "model": "surya", "html": html}, indent=2, ensure_ascii=False))


def run_stage_table_full(client: SuryaClient, input_dir: Path, output_dir: Path, concurrency: int) -> None:
    from tqdm import tqdm

    crops = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not crops:
        print("[peyk-surya] no crop images found in input", file=sys.stderr)
        return

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_process_crop_table_full, crop_path, client, output_dir): crop_path for crop_path in crops}
        for future in tqdm(as_completed(futures), total=len(futures), desc="[peyk-surya] table-full", unit="table", file=sys.stderr):
            crop_path = futures[future]
            try:
                future.result()
            except Exception as e:
                tqdm.write(f"[peyk-surya] {crop_path.name} failed: {e}", file=sys.stderr)
```

- [ ] **Step 2: Add imports and the sharpness threshold constant**

Add to `containers/peyk-surya/run.py`'s imports (near the top, alongside the existing
`from backends.base import ...` / `from backends.client import ...` lines):

```python
import math

from backends.sharpness import laplacian_variance
from backends.table_scale import safe_upscale
from backends.table_split import body_center, center_split_index, max_per_chunk_to_boundaries, split_cols, split_rows
from backends.table_stitch import detect_label_chunk_index, parse_html_table, render_html, stitch_landscape, stitch_portrait
from backends.tsr_dispatch import dispatch_tsr
```

Add alongside `RENDER_SCALE`/`DEFAULT_SERVER_URL` near the top of the file:

```python
# Q1's blur-detection trigger (docs-personal/surya/improvement.md) — do not change without
# re-deriving against that doc's evidence table.
SHARPNESS_THRESHOLD_LAPLACIAN_VAR = 450

# Shown between unstitched fragments when Q4's stitching can't confidently merge split
# chunks — never fail the request, never merge positionally (see Global Constraints).
CONTINUATION_MARKER = "<p>يتبع الجدول التالي من الجدول السابق</p>"
```

- [ ] **Step 3: Replace `_process_crop_table_full` with the smart version**

Replace the entire `_html_from_table_full_result`/`_process_crop_table_full` block (shown in
Step 1) with:

```python
def _html_from_table_full_result(result) -> str:
    """TableRecPredictor.predict_full's result shape is NOT confirmed the way
    RecognitionPredictor's PageOCRResult/BlockOCRResult shape now is (see
    _blocks_from_recognition_result's docstring for that investigation) — this guess
    (flat `.html`/`.text` attribute) hasn't been checked against Surya's real source or a live
    response. Expect to revisit this the same way _text_from_recognition_result's original
    guess had to be, once this stage actually runs against the live server."""
    return getattr(result, "html", None) or getattr(result, "text", "") or ""


def _predict_table_full_html(image, client: SuryaClient) -> str:
    result = client.predict_table_full(image)
    return _html_from_table_full_result(result)


def _split_and_recognize(image, workdir: Path, crop_stem: str, client: SuryaClient) -> str:
    """Q2-Q4: below-threshold correction path. Runs TSR (via peyk-tsr, a new capability — see
    backends/tsr_dispatch.py), splits at the boundary nearest the geometric center, upscales
    each chunk, recognizes each independently, and stitches. Returns HTML — either a clean
    merged table, or (if stitching can't confidently merge) the chunks' own HTML concatenated
    with CONTINUATION_MARKER between them, per this project's "never fail, never merge
    positionally" rule."""
    w, h = image.size

    # dispatch_tsr needs a real file on disk (it copies it into peyk-tsr's input dir) — the
    # caller already has `image` in memory from crop_path, so persist it once under workdir
    # (a real subpath of the shared volume, required by dispatch_tsr's own docstring) before
    # calling TSR.
    scratch_image_path = workdir / f"{crop_stem}_tsr_input.png"
    image.save(scratch_image_path)
    aug = dispatch_tsr(scratch_image_path, workdir)

    axis = "rows" if h >= w else "cols"
    cells = aug.get("cells")

    if axis == "rows":
        rows = sorted(aug["rows"], key=lambda r: r["row"])
        body = rows[1:]
        boundary = center_split_index(body, body_center(body, axis_key="row"), axis_key="row", cells=cells)
        chunk_images_and_indices = split_rows(image, rows, [boundary])
        expected_counts = [len(indices) for _, indices in chunk_images_and_indices]
    else:
        cols = sorted(aug["cols"], key=lambda c: c["col"])
        boundary = center_split_index(cols, w / 2, axis_key="col", cells=cells)
        chunk_images_and_indices = split_cols(image, cols, [boundary])
        expected_counts = [len(indices) for _, indices in chunk_images_and_indices]

    scaled_images = [safe_upscale(chunk_image) for chunk_image, _ in chunk_images_and_indices]
    chunk_htmls = [_predict_table_full_html(chunk_image, client) for chunk_image in scaled_images]
    chunk_grids = [parse_html_table(html) for html in chunk_htmls]

    if axis == "rows":
        final_rows, warnings = stitch_portrait(chunk_grids, expected_counts)
    else:
        label_idx = detect_label_chunk_index(chunk_grids)
        if label_idx is None:
            ordered_grids, ordered_counts = chunk_grids, expected_counts
        else:
            order = [label_idx] + [i for i in range(len(chunk_grids)) if i != label_idx]
            ordered_grids = [chunk_grids[i] for i in order]
            ordered_counts = [expected_counts[i] for i in order]
        final_rows, warnings = stitch_landscape(ordered_grids, ordered_counts)

    if not final_rows:
        # Stitching refused to merge — never fail, never guess: hand back the chunks' own HTML
        # with a continuation marker between them instead.
        print(f"[peyk-surya] {crop_stem}: split chunks could not be stitched ({'; '.join(warnings)}) — returning unstitched with continuation marker", file=sys.stderr)
        return f"\n{CONTINUATION_MARKER}\n".join(chunk_htmls)

    if warnings:
        print(f"[peyk-surya] {crop_stem}: stitched with recoverable warnings: {'; '.join(warnings)}", file=sys.stderr)

    return render_html(final_rows)


def _process_crop_table_full(crop_path: Path, client: SuryaClient, output_dir: Path, workdir: Path) -> None:
    from PIL import Image

    image = Image.open(crop_path).convert("RGB")
    lap_var = laplacian_variance(image)

    if lap_var >= SHARPNESS_THRESHOLD_LAPLACIAN_VAR:
        html = _predict_table_full_html(image, client)
    else:
        print(f"[peyk-surya] {crop_path.stem}: laplacian_var={lap_var:.1f} < {SHARPNESS_THRESHOLD_LAPLACIAN_VAR} — splitting", file=sys.stderr)
        html = _split_and_recognize(image, workdir, crop_path.stem, client)

    out_path = output_dir / f"{crop_path.stem}.json"
    out_path.write_text(json.dumps({"crop": crop_path.name, "model": "surya", "html": html}, indent=2, ensure_ascii=False))
```

- [ ] **Step 4: Update `run_stage_table_full` to pass a real workdir**

Replace `run_stage_table_full` (originally lines 327-346) with:

```python
def run_stage_table_full(client: SuryaClient, input_dir: Path, output_dir: Path, concurrency: int) -> None:
    from tqdm import tqdm

    crops = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not crops:
        print("[peyk-surya] no crop images found in input", file=sys.stderr)
        return

    # workdir for any crop that needs splitting: a real subpath of input_dir's parent, which
    # is itself a subpath of the shared workdir volume this container was launched with
    # --volumes-from ORCHESTRATOR_CONTAINER_NAME for (see backends/tsr_dispatch.py's
    # docstring for why this must be host-visible, not a tempfile.TemporaryDirectory()).
    workdir = input_dir.parent / "table_full_split_workdir"
    workdir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_process_crop_table_full, crop_path, client, output_dir, workdir): crop_path for crop_path in crops}
        for future in tqdm(as_completed(futures), total=len(futures), desc="[peyk-surya] table-full", unit="table", file=sys.stderr):
            crop_path = futures[future]
            try:
                future.result()
            except Exception as e:
                tqdm.write(f"[peyk-surya] {crop_path.name} failed: {e}", file=sys.stderr)
```

- [ ] **Step 5: Verify the image builds**

Run: `cd containers/peyk-surya && docker build -t peyk-surya:dev .`
Expected: build succeeds with no errors.

- [ ] **Step 6: Verify a sharp crop still takes the direct (unsplit) path**

Using an existing sharp table crop (e.g. `hotstorage/workdir/table_full_in/cib_sample__r12.png`,
`laplacian_var` ~1095, well above the 450 threshold) and a running `peyk-vllm-surya`:

```bash
mkdir -p /tmp/peyk_surya_sharp_test/in /tmp/peyk_surya_sharp_test/out
cp hotstorage/workdir/table_full_in/cib_sample__r12.png /tmp/peyk_surya_sharp_test/in/
docker run --rm --network peyk-net \
  -v /tmp/peyk_surya_sharp_test/in:/data/in:ro \
  -v /tmp/peyk_surya_sharp_test/out:/data/out \
  peyk-surya:dev --mode stage --stage table-full --input /data/in --output /data/out
cat /tmp/peyk_surya_sharp_test/out/cib_sample__r12.json
```

Expected: the stderr log shows no `"laplacian_var=... < 450 — splitting"` line for this crop
(confirms the sharp path skipped TSR/splitting entirely, calling `predict_table_full` directly
as before); the output JSON's `html` is a single non-empty `<table>...</table>` string.

- [ ] **Step 7: Verify a blurry crop takes the split path and produces a real merged table**

This requires `peyk-tsr:dev` built (Task 5) and the docker socket available to this container
— run it with the socket mounted the same way `pipeline.py` now does:

```bash
mkdir -p /tmp/peyk_surya_blur_test/in /tmp/peyk_surya_blur_test/out /tmp/peyk_surya_blur_test/workdir
python3 -c "
from PIL import Image, ImageFilter
im = Image.open('hotstorage/workdir/table_full_in/cib_sample__r12.png').convert('RGB')
im.filter(ImageFilter.GaussianBlur(radius=1)).save('/tmp/peyk_surya_blur_test/in/cib_r12_blurred.png')
"
docker rm -f peyk-orchestrator-run 2>/dev/null || true
docker run -d --name peyk-orchestrator-run --network peyk-net -v /tmp/peyk_surya_blur_test/workdir:/workdir alpine sleep 3600
docker run --rm --network peyk-net \
  --volumes-from peyk-orchestrator-run \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /tmp/peyk_surya_blur_test/in:/data/in:ro \
  -v /tmp/peyk_surya_blur_test/out:/data/out \
  peyk-surya:dev --mode stage --stage table-full --input /data/in --output /data/out
cat /tmp/peyk_surya_blur_test/out/cib_r12_blurred.json
docker rm -f peyk-orchestrator-run
```

Expected: stderr shows a `laplacian_var=... < 450 — splitting` line for this crop; the output
JSON's `html` is one merged `<table>` (not two, not a `CONTINUATION_MARKER` — this specific
crop/blur level was already validated end-to-end in `z_surya_exploration` to stitch cleanly at
the default center-split boundary).

- [ ] **Step 8: Commit**

```bash
git add containers/peyk-surya/run.py
git commit -m "feat(peyk-surya): wire Q1-Q4 smart split/stitch into --stage table-full"
```

---

### Task 8: Documentation update

**Files:**
- Modify: `docs-personal/surya/improvement.md`

- [ ] **Step 1: Mark the integration as done**

In the `## Before integrating into peyk-surya/peyk-orchestrator` section, update the
introductory sentence and each of items 1-4 (the ones with code, not the document-only items
5-8) to note they're now implemented, referencing this plan file and the new modules:
`containers/peyk-surya/backends/{sharpness,table_split,table_scale,table_stitch,tsr_dispatch}.py`,
the `run.py` wiring, and the `Dockerfile`/`pipeline.py`/`stages.py` docker-socket changes.

- [ ] **Step 2: Commit**

```bash
git add docs-personal/surya/improvement.md
git commit -m "docs: mark peyk-surya smart table-full integration as implemented"
```
