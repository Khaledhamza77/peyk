from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Region:
    """A detected layout region on a page."""

    page: int
    label: str  # one of: text, table, figure (backend-specific labels get mapped to these)
    score: float
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1 in pixel coords

    def to_dict(self) -> dict:
        return {
            "page": self.page,
            "label": self.label,
            "score": self.score,
            "bbox": list(self.bbox),
        }


class LayoutBackend(ABC):
    """Common interface every layout backend must implement."""

    name: str

    @abstractmethod
    def load(self) -> None:
        """Load model weights. Called once before predict()."""

    @abstractmethod
    def predict(self, image_path: Path) -> list[Region]:
        """Run layout detection on a single page image, return detected regions."""


# Found empirically on heron.py's model (see that module): class-agnostic, not per-label —
# the observed duplicates weren't consistently same-label (e.g. a "figure" and "text" box
# sharing an identical bbox), so restricting suppression to same-label pairs would miss
# exactly that case. Shared here (not left heron-only) so any backend can apply the same
# safety net — a backend with no duplicate-box problem of its own pays no real cost, since
# non-overlapping boxes are untouched either way.
NMS_IOU_THRESHOLD = 0.5

# Plain IoU (intersection/union) misses full-containment duplicates when the two boxes are
# very different sizes: e.g. a large mis-merged box that happens to fully enclose a smaller,
# correctly-sized one — union is dominated by the large box's area, so IoU stays low (an
# observed real case: ~0.12) even though the smaller box is 100% inside the larger one.
# Intersection-over-minimum-area (IoM) instead scores full containment as 1.0 regardless of
# the size ratio between the two boxes, catching exactly this case.
NMS_CONTAINMENT_THRESHOLD = 0.8


def _areas_and_intersection(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float, float]:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    return area_a, area_b, inter


def _overlaps(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    iou_threshold: float,
    containment_threshold: float,
) -> bool:
    area_a, area_b, inter = _areas_and_intersection(a, b)
    if inter == 0.0:
        return False
    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 0.0
    iom = inter / min(area_a, area_b) if min(area_a, area_b) > 0 else 0.0
    return iou > iou_threshold or iom > containment_threshold


def nms(
    regions: list[Region],
    iou_threshold: float = NMS_IOU_THRESHOLD,
    containment_threshold: float = NMS_CONTAINMENT_THRESHOLD,
) -> list[Region]:
    """Class-agnostic non-max suppression (by IoU) plus containment suppression (by IoM) —
    see the two threshold constants above for why both are needed. Greedy: keeps the
    highest-scoring region in each overlapping/contained cluster."""
    kept: list[Region] = []
    for region in sorted(regions, key=lambda r: r.score, reverse=True):
        if any(_overlaps(region.bbox, k.bbox, iou_threshold, containment_threshold) for k in kept):
            continue
        kept.append(region)
    return kept
