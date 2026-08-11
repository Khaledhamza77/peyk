from pathlib import Path

from .base import LayoutBackend, Region, nms

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
# labels, e.g. a "figure" and "text" box sharing an identical bbox). base.py's nms() is
# class-agnostic for exactly this reason — the observed duplicates weren't consistently
# same-label, so restricting suppression to same-label pairs would have missed the
# cross-label case above. Its default thresholds were tuned against this backend's observed
# behavior specifically (see base.py) and are reused as-is by the other layout backends too.


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
        return nms(regions)
