from pathlib import Path

from .base import Cell, TSRBackend, TableStructure

_REPO_ID = "ds4sd/docling-models"
_VARIANT = "accurate"  # docling-ibm-models also ships a "fast" variant; accurate chosen as default
# ds4sd/docling-models bundles artifacts for every Docling model family (layout, picture
# classifier, tableformer, ...), not just this one — scoped to only the tableformer/accurate
# subtree so build-time baking (see Dockerfile) doesn't pull down unrelated multi-GB assets.
# No revision pin: left at the repo's default branch rather than a guessed tag, since an
# invalid pin would hard-fail the Dockerfile's bake step immediately.
_ALLOW_PATTERNS = [f"model_artifacts/tableformer/{_VARIANT}/*"]


class TableFormerBackend(TSRBackend):
    """`predict_details["table_cells"]` gives already-structured per-cell entries directly
    (row_id/column_id/bbox, plus top-level num_rows/num_cols) — confirmed by introspecting a
    real prediction, no HTML-token parsing needed here (unlike pp_structure.py)."""

    name = "tableformer"

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._predictor = None

    def load(self) -> None:
        from huggingface_hub import snapshot_download
        from docling_ibm_models.tableformer.data_management.tf_predictor import TFPredictor
        import docling_ibm_models.tableformer.common as tf_common

        artifact_path = (
            Path(snapshot_download(repo_id=_REPO_ID, allow_patterns=_ALLOW_PATTERNS))
            / "model_artifacts"
            / "tableformer"
            / _VARIANT
        )
        config = tf_common.read_config(str(artifact_path / "tm_config.json"))
        config["model"]["save_dir"] = str(artifact_path)
        self._predictor = TFPredictor(config, device=self.device, num_threads=4)

    def predict(self, image_path: Path) -> TableStructure:
        if self._predictor is None:
            raise RuntimeError("Backend not loaded; call load() first")

        import numpy as np
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        # do_matching=False: skip OCR-token-to-cell content matching entirely — this backend
        # only needs the predicted cell bboxes/structure, peyk-orchestrator pairs text in
        # separately (per-cell, via peyk-dcr/peyk-ocr), same as every other backend here.
        iocr_page = {"image": np.array(image), "tokens": [], "width": width, "height": height}
        table_bbox = [0, 0, width, height]
        tf_output = self._predictor.multi_table_predict(iocr_page, [table_bbox], do_matching=False)

        details = tf_output[0]["predict_details"]
        # Spanning cells (multicol_tag set) aren't reconstructed into multi-cell spans —
        # every cell emitted as row_span=col_span=1, same simplification as
        # grid_from_rows_and_cols in base.py.
        cells = [
            Cell(
                row=c["row_id"],
                col=c["column_id"],
                row_span=1,
                col_span=1,
                bbox=tuple(float(v) for v in c["bbox"]),
            )
            for c in details["table_cells"]
        ]
        return TableStructure(num_rows=details["num_rows"], num_cols=details["num_cols"], cells=cells)
