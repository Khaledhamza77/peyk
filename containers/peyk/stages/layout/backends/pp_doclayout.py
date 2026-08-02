from pathlib import Path

from .base import LayoutBackend, Region, nms

# PP-DocLayoutV2 groups regions into many fine-grained classes (title, paragraph_title,
# text, table, figure/image, chart, formula, seal/stamp, ...). We collapse them down to
# the three classes the rest of the pipeline branches on: text / table / figure.
_LABEL_MAP = {
    "paragraph_title": "text",
    "doc_title": "text",
    "text": "text",
    "abstract": "text",
    "content": "text",
    "reference": "text",
    "footnote": "text",
    "header": "text",
    "footer": "text",
    "number": "text",
    "formula": "text",
    "table": "table",
    "table_title": "text",
    "image": "figure",
    "figure": "figure",
    "figure_title": "text",
    "chart": "figure",
    "chart_title": "text",
    "seal": "figure",
    "algorithm": "text",
}


class PPDocLayoutV2Backend(LayoutBackend):
    name = "pp-doclayout-v2"

    def __init__(self, device: str = "gpu"):
        self.device = device
        self._model = None

    def load(self) -> None:
        from paddlex import create_model

        self._model = create_model(model_name="PP-DocLayoutV2", device=self.device)

    def predict(self, image_path: Path) -> list[Region]:
        if self._model is None:
            raise RuntimeError("Backend not loaded; call load() first")

        regions: list[Region] = []
        for result in self._model.predict(str(image_path), batch_size=1):
            for box in result.get("boxes", []):
                raw_label = box["label"]
                label = _LABEL_MAP.get(raw_label, "text")
                x0, y0, x1, y1 = box["coordinate"]
                regions.append(
                    Region(
                        page=0,
                        label=label,
                        score=float(box["score"]),
                        bbox=(float(x0), float(y0), float(x1), float(y1)),
                    )
                )
        # See doclayout_yolo.py's predict() for why this is applied here even without
        # independent confirmation this backend has Heron's duplicate-box problem.
        return nms(regions)
