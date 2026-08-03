from pathlib import Path

from .base import TSRBackend, TableStructure, grid_from_rows_and_cols

_MODEL_ID = "microsoft/table-transformer-structure-recognition"

# TATR's own id2label for this checkpoint. Only rows/columns feed the grid builder here;
# "table"/"table column header"/"table spanning cell"/"table projected row header" are
# ignored — see grid_from_rows_and_cols's spanning-cell simplification note.
_ROW_LABEL = "table row"
_COL_LABEL = "table column"

_SCORE_THRESHOLD = 0.7


class TATRBackend(TSRBackend):
    name = "tatr"

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._processor = None
        self._model = None

    def load(self) -> None:
        import torch
        from transformers import AutoImageProcessor, TableTransformerForObjectDetection

        self._processor = AutoImageProcessor.from_pretrained(_MODEL_ID)
        self._model = TableTransformerForObjectDetection.from_pretrained(_MODEL_ID).to(self.device)
        self._model.eval()
        self._torch = torch

    def predict(self, image_path: Path) -> TableStructure:
        if self._model is None:
            raise RuntimeError("Backend not loaded; call load() first")

        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = self._torch.tensor([image.size[::-1]])
        result = self._processor.post_process_object_detection(
            outputs, threshold=_SCORE_THRESHOLD, target_sizes=target_sizes
        )[0]

        id2label = self._model.config.id2label
        rows, cols = [], []
        for label_id, box in zip(result["labels"].tolist(), result["boxes"].tolist()):
            label = id2label[label_id]
            if label == _ROW_LABEL:
                rows.append(tuple(box))
            elif label == _COL_LABEL:
                cols.append(tuple(box))

        return grid_from_rows_and_cols(rows, cols)
