from pathlib import Path

from .base import LayoutBackend, Region

# DocLayNet-style class names (Docling's layout taxonomy) -> the three classes the rest
# of the pipeline branches on.
_LABEL_MAP = {
    "Caption": "text",
    "Footnote": "text",
    "Formula": "text",
    "List-item": "text",
    "Page-footer": "text",
    "Page-header": "text",
    "Picture": "figure",
    "Section-header": "text",
    "Table": "table",
    "Text": "text",
    "Title": "text",
    "Document Index": "text",
    "Code": "text",
    "Checkbox-Selected": "text",
    "Checkbox-Unselected": "text",
    "Form": "text",
    "Key-Value Region": "text",
}

_REPO_ID = "docling-project/docling-layout-heron"

# LayoutPredictor's own default (0.3) let through a lot of low-confidence noise that isn't
# a near-duplicate of anything else (so NMS below doesn't touch it either) — e.g. 0.30-0.32
# boxes sitting in otherwise-empty space in the raw cib_sample.pdf output. This is a
# complementary fix, not a replacement for NMS: it drops spurious low-confidence detections,
# NMS drops duplicate/contained detections of a genuinely good one — a low-confidence box
# can still legitimately overlap a high-confidence one and need NMS, and a non-overlapping
# low-confidence box needs this threshold instead, since NMS has nothing to compare it against.
_CONFIDENCE_THRESHOLD = 0.5

# docling_ibm_models's LayoutPredictor.predict() only applies HF's
# post_process_object_detection(..., threshold=...) — a confidence-score filter, no NMS
# (DETR-family models are architecturally supposed to be NMS-free, but empirically this one
# isn't: raw output on cib_sample.pdf included near-duplicate boxes even across different
# labels, e.g. a "figure" and "text" box sharing an identical bbox). Class-agnostic, not
# per-label: the observed duplicates weren't consistently same-label, so restricting
# suppression to same-label pairs would have missed exactly the cross-label case above.
_NMS_IOU_THRESHOLD = 0.5

# Plain IoU (intersection/union) misses full-containment duplicates when the two boxes are
# very different sizes: e.g. a large mis-merged box that happens to fully enclose a smaller,
# correctly-sized one — union is dominated by the large box's area, so IoU stays low (an
# observed real case: ~0.12) even though the smaller box is 100% inside the larger one.
# Intersection-over-minimum-area (IoM) instead scores full containment as 1.0 regardless of
# the size ratio between the two boxes, catching exactly this case.
_NMS_CONTAINMENT_THRESHOLD = 0.8


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


def _nms(
    regions: list[Region],
    iou_threshold: float = _NMS_IOU_THRESHOLD,
    containment_threshold: float = _NMS_CONTAINMENT_THRESHOLD,
) -> list[Region]:
    kept: list[Region] = []
    for region in sorted(regions, key=lambda r: r.score, reverse=True):
        if any(_overlaps(region.bbox, k.bbox, iou_threshold, containment_threshold) for k in kept):
            continue
        kept.append(region)
    return kept


class HeronBackend(LayoutBackend):
    name = "heron"

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._predictor = None

    def load(self) -> None:
        from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor
        from huggingface_hub import snapshot_download

        artifact_path = snapshot_download(repo_id=_REPO_ID)
        self._predictor = LayoutPredictor(
            artifact_path, device=self.device, base_threshold=_CONFIDENCE_THRESHOLD
        )

    def predict(self, image_path: Path) -> list[Region]:
        if self._predictor is None:
            raise RuntimeError("Backend not loaded; call load() first")

        from PIL import Image

        image = Image.open(image_path)
        regions: list[Region] = []
        for pred in self._predictor.predict(image):
            label = _LABEL_MAP.get(pred["label"], "text")
            regions.append(
                Region(
                    page=0,
                    label=label,
                    score=float(pred["confidence"]),
                    bbox=(float(pred["l"]), float(pred["t"]), float(pred["r"]), float(pred["b"])),
                )
            )
        return _nms(regions)
