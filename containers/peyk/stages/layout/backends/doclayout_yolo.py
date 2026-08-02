from pathlib import Path

from .base import LayoutBackend, Region, nms

# DocStructBench class names -> the three classes the rest of the pipeline branches on.
_LABEL_MAP = {
    "title": "text",
    "plain text": "text",
    "abandon": "text",
    "figure": "figure",
    "figure_caption": "text",
    "table": "table",
    "table_caption": "text",
    "table_footnote": "text",
    "isolate_formula": "text",
    "formula_caption": "text",
}

_REPO_ID = "juliozhao/DocLayout-YOLO-DocStructBench"
_FILENAME = "doclayout_yolo_docstructbench_imgsz1024.pt"


class DocLayoutYOLOBackend(LayoutBackend):
    name = "doclayout-yolo"

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self._model = None

    def load(self) -> None:
        from doclayout_yolo import YOLOv10
        from huggingface_hub import hf_hub_download

        weights_path = hf_hub_download(repo_id=_REPO_ID, filename=_FILENAME)
        self._model = YOLOv10(weights_path)

    def predict(self, image_path: Path) -> list[Region]:
        if self._model is None:
            raise RuntimeError("Backend not loaded; call load() first")

        results = self._model.predict(str(image_path), imgsz=1024, conf=0.2, device=self.device)
        result = results[0]
        names = result.names

        regions: list[Region] = []
        for box in result.boxes:
            raw_label = names[int(box.cls[0])]
            label = _LABEL_MAP.get(raw_label, "text")
            x0, y0, x1, y1 = (float(v) for v in box.xyxy[0].tolist())
            regions.append(
                Region(
                    page=0,
                    label=label,
                    score=float(box.conf[0]),
                    bbox=(x0, y0, x1, y1),
                )
            )
        # Not independently confirmed to have Heron's duplicate-box problem (this backend
        # isn't the picked default, so it hasn't seen the same real-document scrutiny) — applied
        # as a safety net regardless, using base.py's thresholds as-is: a clean, non-overlapping
        # prediction set is untouched by this either way, so there's no real cost if it turns
        # out not to be needed here.
        return nms(regions)
