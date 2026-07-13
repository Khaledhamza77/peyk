"""Thin wrapper around Surya's predictor classes, all backed by one shared
SuryaInferenceManager pointed at the peyk-vllm-surya server (see that container's README)
instead of Surya's own default of auto-spawning a local server — mirrors the wiring in
docs/surya/details.md.

**API shape status, per predictor**: `predict_recognition`'s result shape (`List[PageOCRResult]`,
each with a `.blocks` list of `BlockOCRResult`) is CONFIRMED against Surya's real source
(surya/recognition/__init__.py) — see run.py's `_blocks_from_recognition_result`. An earlier
flat `.html`/`.confidence` guess directly on the result was wrong and silently produced empty
OCR output every time; don't reintroduce that shape. `predict_table_full`'s result shape is
ALSO confirmed (surya/table_rec/__init__.py): `predict_full(images, counts=None) ->
List[TableResult]`, one flat `TableResult` per image (unlike recognition's nested
per-block list) with a genuine `.html` attribute — the original flat-attribute guess for this
one happened to be right, but the real, separate bug was `counts` being left unset (see
predict_table_full's own docstring below for the full story — every real table came back
truncated because of it, unrelated to the guessed shape being correct or not).
`predict_layout`/`predict_table_structure` are still UNCONFIRMED guesses (`.bboxes`/`.label`/
`.confidence`/`.bbox` for layout; `cell.row_id`/`cell.col_id`/`.rowspan`/`.colspan`/`.bbox` for
table structure) — expect to revisit each the same way recognition's guess had to be, see e.g.
peyk-layout/backends/heron.py's _LABEL_MAP for the same kind of empirical-tuning step every
other backend's label map/result parsing in this project went through."""
import os

_DEFAULT_INFERENCE_URL = "http://peyk-vllm-surya:8000/v1"

# Best-effort placeholder, built from Surya's publicly documented layout taxonomy (close to
# the DocLayNet-style categories docling-layout-heron also uses — see peyk-layout/backends/
# heron.py's _LABEL_MAP for the same mapping applied to a different model's labels). NOT yet
# verified against a real Surya-OCR-2 prediction; update once one is available, same as every
# other layout backend's label map in this project.
LABEL_MAP = {
    "Caption": "text",
    "Footnote": "text",
    "Formula": "text",
    "List-item": "text",
    "Page-footer": "text",
    "Page-header": "text",
    "Picture": "figure",
    "Figure": "figure",
    "Section-header": "text",
    "Table": "table",
    "Text": "text",
    "TextInlineMath": "text",
    "Title": "text",
    "TableOfContents": "text",
    "Handwriting": "text",
    "Code": "text",
    "Form": "text",
}


class SuryaClient:
    """Loads only the predictor roles actually requested (`roles`), so a --stage layout
    invocation doesn't pay for constructing a TableRecPredictor it'll never use, etc."""

    def __init__(self, server_url: str | None = None):
        self.server_url = server_url or _DEFAULT_INFERENCE_URL
        self._manager = None
        self._layout = None
        self._recognition = None
        self._table_rec = None

    def load(self, roles: set[str]) -> None:
        os.environ["SURYA_INFERENCE_URL"] = self.server_url
        os.environ["SURYA_INFERENCE_BACKEND"] = "vllm"

        from surya.inference import SuryaInferenceManager

        self._manager = SuryaInferenceManager()

        if "layout" in roles:
            from surya.layout import LayoutPredictor

            self._layout = LayoutPredictor(self._manager)
        if "recognition" in roles:
            from surya.recognition import RecognitionPredictor

            self._recognition = RecognitionPredictor(self._manager)
        if "table_rec" in roles:
            from surya.table_rec import TableRecPredictor

            self._table_rec = TableRecPredictor(self._manager)

    def predict_layout(self, image):
        """Returns Surya's raw per-image layout result. Caller (run.py) maps it into this
        project's Region dataclass — kept separate so the "unconfirmed shape" risk is isolated
        to one call site per role instead of spread across run.py."""
        if self._layout is None:
            raise RuntimeError("layout role not loaded; pass 'layout' to load(roles=...)")
        return self._layout([image])[0]

    def predict_recognition(self, image, layout_result=None):
        if self._recognition is None:
            raise RuntimeError("recognition role not loaded; pass 'recognition' to load(roles=...)")
        if layout_result is not None:
            return self._recognition([image], [layout_result])[0]
        return self._recognition([image])[0]

    def predict_table_structure(self, image):
        if self._table_rec is None:
            raise RuntimeError("table_rec role not loaded; pass 'table_rec' to load(roles=...)")
        return self._table_rec([image])[0]

    # predict_full's own signature is predict_full(images, counts=None) — counts is a
    # per-image hint used ONLY to size max_tokens (image_token_budget(count, ceiling=
    # SURYA_MAX_TOKENS_BLOCK_CEILING, floor=1024), confirmed against surya/table_rec/
    # __init__.py's real source: clamp(count + 100, floor, ceiling)). Leaving counts unset
    # defaults it to 0 per image, clamping the budget down to the floor (1024) regardless of
    # the table's real size — confirmed live: every real table sent through predict_full came
    # back truncated, and raising SURYA_MAX_TOKENS_TABLE_REC (a *different* method's setting,
    # not this one) had zero effect, exactly as this code explains it should.
    #
    # _LARGE_COUNT is a deliberate, generous stand-in for a real per-table cell count, not a
    # genuine estimate — getting a real count means calling predict_table_structure first
    # (the only predictor that actually knows row/col/cell counts), which would turn
    # predict_full's whole "one call per table" design back into two calls, undoing the exact
    # efficiency win it exists for. Passing a large count instead just reliably clamps to the
    # ceiling (SURYA_MAX_TOKENS_BLOCK_CEILING, tuned down via env var to stay under
    # peyk-vllm-surya's --max-model-len=8000 — see Dockerfile) — a legitimate use of a
    # documented sizing hint, not a hack around any safety check. Revisit only if this wastes
    # meaningful KV-cache budget once running under real concurrency.
    _LARGE_COUNT = 9999

    def predict_table_full(self, image):
        if self._table_rec is None:
            raise RuntimeError("table_rec role not loaded; pass 'table_rec' to load(roles=...)")
        return self._table_rec.predict_full([image], counts=[self._LARGE_COUNT])[0]
