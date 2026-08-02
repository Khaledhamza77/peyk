from pathlib import Path

from .base import TSRBackend, TableStructure, grid_from_html_tokens

# SLANet_plus is PaddleX's general-purpose table-structure-recognition module used by the
# PP-StructureV3 pipeline (this backend calls the module directly via create_model, same
# pattern as peyk-layout's PP-DocLayoutV2 backend, rather than the full PP-StructureV3
# pipeline object — we only want the table-structure step, not PP-StructureV3's own
# layout+OCR bundling). One model regardless of the table's own wired/wireless styling.
_GENERAL_MODEL_NAME = "SLANet_plus"

# PP-LCNet_x1_0_table_cls: PaddleX's own lightweight (MobileNet-family) wired/wireless table
# classifier — confirmed via a real run to output label_names ["wired_table", "wireless_table"].
# This is the same classify-then-route step PP-StructureV3's own pipeline uses internally,
# rather than picking one general-purpose structure model for every table regardless of style.
_CLASSIFIER_MODEL_NAME = "PP-LCNet_x1_0_table_cls"
_WIRED_MODEL_NAME = "SLANeXt_wired"
_WIRELESS_MODEL_NAME = "SLANeXt_wireless"


def _predict_structure(model, image_path: Path) -> TableStructure:
    for result in model.predict(str(image_path), batch_size=1):
        # Field names ("bbox"/"structure") match PaddleX's table-structure-recognition
        # module as of paddlex==3.7.0 (same version pinned for PP-DocLayoutV2) — verified
        # against a real result dict.
        return grid_from_html_tokens(result["structure"], result["bbox"])
    return TableStructure(num_rows=0, num_cols=0, cells=[])


class PPStructureGeneralBackend(TSRBackend):
    """Single general-purpose model (SLANet_plus), no wired/wireless routing — see
    PPStructureWiringBackend below for the classify-then-route alternative."""

    name = "pp-structure-general"

    def __init__(self, device: str = "gpu"):
        self.device = device
        self._model = None

    def load(self) -> None:
        from paddlex import create_model

        self._model = create_model(model_name=_GENERAL_MODEL_NAME, device=self.device)

    def predict(self, image_path: Path) -> TableStructure:
        if self._model is None:
            raise RuntimeError("Backend not loaded; call load() first")
        return _predict_structure(self._model, image_path)


class PPStructureWiringBackend(TSRBackend):
    """Classifies each crop wired vs wireless first (PP-LCNet_x1_0_table_cls), then routes
    to the matching structure model (SLANeXt_wired / SLANeXt_wireless) — the same two-stage
    approach PP-StructureV3's own pipeline uses internally, instead of one general-purpose
    model for every table regardless of border styling. Both structure models are loaded
    eagerly at load() time (not lazily per first use) so a single container instance never
    pays a mid-batch model-load latency spike on the first crop of whichever style it hasn't
    seen yet."""

    name = "pp-structure-wiring"

    def __init__(self, device: str = "gpu"):
        self.device = device
        self._classifier = None
        self._wired_model = None
        self._wireless_model = None

    def load(self) -> None:
        from paddlex import create_model

        self._classifier = create_model(model_name=_CLASSIFIER_MODEL_NAME, device=self.device)
        self._wired_model = create_model(model_name=_WIRED_MODEL_NAME, device=self.device)
        self._wireless_model = create_model(model_name=_WIRELESS_MODEL_NAME, device=self.device)

    def predict(self, image_path: Path) -> TableStructure:
        if self._classifier is None:
            raise RuntimeError("Backend not loaded; call load() first")

        for result in self._classifier.predict(str(image_path), batch_size=1):
            # label_names/scores are both ordered by descending score (confirmed via a real
            # run) — label_names[0] is always the top prediction, not necessarily "wired_table".
            label = result["label_names"][0]
            break
        else:
            label = "wired_table"

        model = self._wired_model if label == "wired_table" else self._wireless_model
        return _predict_structure(model, image_path)
