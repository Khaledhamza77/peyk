"""Pipeline config: which model/image each job uses. Every job's model must be explicitly
configured — there's no "stub" concept anymore (that was only ever a placeholder for a stage
whose container hadn't been built yet; all seven now exist)."""
import functools
from dataclasses import dataclass
from pathlib import Path

import yaml

from stages import list_vlm_models


# Which image each classical ocr model runs in — see containers/peyk-simple-ocr/ and
# containers/peyk-paddleocr-vl/ (the split rationale is in implementation_plan.md Task 1.3).
# Derived automatically from `model` rather than requiring `image` in the yaml too: the
# pairing is fully determined by the model choice, so making both configurable independently
# just invites a config that sets model: paddleocr-vl but forgets to also flip image (or vice
# versa) — a mismatch `run_docker_stage` has no way to catch, since it just docker-runs
# whatever image/model string it's given.
OCR_MODEL_IMAGES = {
    "paddleocr-vl": "peyk-paddleocr-vl:dev",
    "paddleocr": "peyk-simple-ocr:dev",
    "easyocr": "peyk-simple-ocr:dev",
    "rapidocr": "peyk-simple-ocr:dev",
    "tesseract": "peyk-simple-ocr:dev",
    "surya": "peyk-surya:dev",
}

# Same derivation for layout. "surya-layout" (peyk-layout's own unimplemented stub backend,
# unrelated to this project's real Surya integration) is deliberately not included here: it
# would raise NotImplementedError at runtime if selected, so it shouldn't be silently blessed
# with an image mapping as if it were a real option. layout has no peyk-vlm option at all —
# no VLM in this project does region detection/layout analysis.
LAYOUT_MODEL_IMAGES = {
    "pp-doclayout-v2": "peyk-layout:dev",
    "doclayout-yolo": "peyk-layout:dev",
    "heron": "peyk-layout:dev",
    "surya": "peyk-surya:dev",
}

# tsr (table STRUCTURE recognition, no text) similarly has no bare peyk-vlm option — every
# peyk-vlm model's "table" role always does structure+text together in one call (see its
# prompts.py), there's no way to invoke it for structure alone. Surya genuinely supports both
# (a real structure-only TableRecPredictor stage, and predict_full for structure+text) so it's
# valid here on its own; a peyk-vlm model is only ever valid for tsr when cell_ocr resolves to
# that exact same model too (full_table_backend below) — enforced by _validate_tsr_and_cell_ocr.
TSR_MODEL_IMAGES = {
    "tatr": "peyk-tsr:dev",
    "rapidtable": "peyk-tsr:dev",
    "pp-structure-general": "peyk-tsr:dev",
    "pp-structure-wiring": "peyk-tsr:dev",
    "tableformer": "peyk-tsr:dev",
    "surya": "peyk-surya:dev",
}

# peyk-paddleocr-vl's default --server-url already matches this (see that container's
# run.py), so this only matters if peyk-vllm-paddleocr is reachable at a different address.
DEFAULT_VLLM_SERVER_URL = "http://peyk-vllm-paddleocr:8118/v1"

# Same reasoning, for peyk-surya against peyk-vllm-surya (see that container's start.sh/README).
DEFAULT_SURYA_SERVER_URL = "http://peyk-vllm-surya:8000/v1"

# peyk-vlm is one image serving many models via --model (see containers/peyk-vlm/backends/
# registry.py) — unlike OCR_MODEL_IMAGES/TSR_MODEL_IMAGES's per-model dict, any model peyk-vlm
# itself reports supporting (see _vlm_models below) derives this single image. Used by
# ocr/cell_ocr, tsr (the full-table combination only), figures, and fullpage alike.
DEFAULT_VLM_IMAGE = "peyk-vlm:dev"

# peyk-dcr has no model concept at all (one approach, pure pypdfium2 extraction) — its image is
# always this, never configurable via yaml. No job's image should ever need to appear in
# example.yaml; every one is either derived from model (above, and OCR_MODEL_IMAGES/
# LAYOUT_MODEL_IMAGES/TSR_MODEL_IMAGES) or, for dcr, simply fixed.
DEFAULT_DCR_IMAGE = "peyk-dcr:dev"


@functools.lru_cache(maxsize=1)
def _vlm_models() -> dict[str, str]:
    """{model_key: provider} for every model peyk-vlm actually supports, queried directly from
    the container (`docker run --rm peyk-vlm:dev --list-models`) rather than assumed from a
    naming convention (e.g. "does this string start with bedrock-/vertex-?") — a convention
    that could silently drift out of sync with the real registry, and offered no way to
    validate a typo'd model name until peyk-vlm's own argparse rejected it at dispatch time.
    Cached: load_config() calls into this multiple times (tsr/ocr/cell_ocr/figures/fullpage),
    and the answer can't change within one process's lifetime."""
    return list_vlm_models(DEFAULT_VLM_IMAGE)


def is_vlm_model(model: str | None) -> bool:
    return model is not None and model in _vlm_models()


def vlm_provider(model: str) -> str:
    """The real provider ("bedrock", "vertex-gemini", "vertex-maas") for a peyk-vlm model key,
    per peyk-vlm's own registry — used to decide which cloud's credentials to mount (see
    pipeline.py's _vlm_credential_docker_args), instead of guessing from the key's spelling."""
    return _vlm_models()[model]


@dataclass
class StageConfig:
    image: str | None = None
    backend: str | None = None  # the chosen model's name (yaml field is called `model`)
    lang: str = "arabic"
    server_url: str | None = None  # ocr only: peyk-paddleocr-vl's/peyk-surya's vLLM-style server URL


@dataclass
class PipelineConfig:
    layout: StageConfig
    ocr: StageConfig
    dcr: StageConfig
    tsr: StageConfig
    figures: StageConfig
    # None (default) => table cells share `ocr`'s model/dispatch, exactly as before. Set this to
    # route isolated table-cell crops to a different model than whole-region text — e.g. keeping
    # cells on a classical model (paddleocr/tesseract) while `ocr.model` is a single-image VLM
    # recognizer (surya/paddleocr-vl/a peyk-vlm model), which reliably mis-recognizes tiny,
    # context-free cell content regardless of which one it is — see implementation_plan.md
    # Task 1.5's open item. Not a "stub" — this is a genuinely optional override, not a
    # placeholder for an unbuilt container.
    cell_ocr: StageConfig | None = None
    born_digital_min_chars: int = 20
    force_scanned: bool = False
    # None (default): the normal per-region layout->tsr->ocr->figures->assembly path. Set to
    # bypass all of that entirely and run one whole-page-at-a-time model over every page
    # instead — model "surya" renders PDFs itself (peyk-surya's own --mode fullpage); any
    # peyk-vlm model key needs pages rendered first (run.py's render_pdf_pages(), since peyk-vlm
    # has no PDF-rendering capability of its own) then dispatches --role fullpage. See run.py.
    fullpage: StageConfig | None = None


def _stage(raw: dict) -> StageConfig:
    return StageConfig(
        image=raw.get("image"),
        backend=raw.get("model"),
        lang=raw.get("lang", "arabic"),
        server_url=raw.get("server_url"),
    )


def _stage_with_image(raw: dict, image_map: dict[str, str]) -> StageConfig:
    """Plain dict-only image derivation — deliberately does NOT fall back to DEFAULT_VLM_IMAGE
    for an unrecognized model, unlike _stage_with_image_or_vlm below. Used for layout, which
    must never resolve a peyk-vlm model to an image at all (no VLM in this project does region
    detection) — leaving stage.image as None for any unrecognized model, including a
    peyk-vlm-shaped one, is exactly what lets _layout_stage's own check below catch and reject
    it with a clear error instead of silently treating it as valid."""
    stage = _stage(raw)
    if stage.image is not None:
        return stage
    if stage.backend in image_map:
        stage.image = image_map[stage.backend]
    return stage


def _stage_with_image_or_vlm(raw: dict, image_map: dict[str, str]) -> StageConfig:
    """Like _stage_with_image, but also derives DEFAULT_VLM_IMAGE for any model peyk-vlm itself
    reports supporting (is_vlm_model, real registry check — see _vlm_models) — used by tsr/ocr,
    which DO have a legitimate (if constrained — see _validate_tsr_and_cell_ocr) peyk-vlm
    option, unlike layout."""
    stage = _stage_with_image(raw, image_map)
    if stage.image is None and is_vlm_model(stage.backend):
        stage.image = DEFAULT_VLM_IMAGE
    return stage


def _layout_stage(raw: dict) -> StageConfig:
    stage = _stage_with_image(raw, LAYOUT_MODEL_IMAGES)
    if stage.image is None:
        raise ValueError(
            f"layout.model {stage.backend!r} is not a recognized layout model "
            f"({sorted(LAYOUT_MODEL_IMAGES)}) — layout can never be done by a peyk-vlm model, "
            "no VLM in this project does region detection."
        )
    return stage


def _tsr_stage(raw: dict) -> StageConfig:
    stage = _stage_with_image_or_vlm(raw, TSR_MODEL_IMAGES)
    if stage.image is None:
        raise ValueError(
            f"tsr.model {stage.backend!r} is not a recognized tsr model "
            f"({sorted(TSR_MODEL_IMAGES)}) or a peyk-vlm model key."
        )
    return stage


def _figures_stage(raw: dict) -> StageConfig:
    """figures: (renamed from the old standalone vlm: section) — describes non-text/non-table
    regions (chart/stamp/photo). peyk-vlm only, no classical equivalent exists, so there's no
    separate "which family of model" choice to make the way ocr/tsr have — just one model key,
    validated against peyk-vlm's own registry (see _vlm_models)."""
    stage = _stage(raw)
    if stage.backend is not None and not is_vlm_model(stage.backend):
        raise ValueError(
            f"figures.model {stage.backend!r} is not a recognized peyk-vlm model "
            f"({sorted(_vlm_models())})."
        )
    if stage.image is None and stage.backend is not None:
        stage.image = DEFAULT_VLM_IMAGE
    return stage


def _fullpage_stage(raw: dict | None) -> StageConfig | None:
    if raw is None:
        return None
    stage = _stage(raw)
    if stage.backend != "surya" and not is_vlm_model(stage.backend):
        raise ValueError(
            f"fullpage.model {stage.backend!r} is not 'surya' or a recognized peyk-vlm model "
            f"({sorted(_vlm_models())})."
        )
    if stage.image is None:
        stage.image = "peyk-surya:dev" if stage.backend == "surya" else DEFAULT_VLM_IMAGE
    return stage


def _dcr_stage(raw: dict) -> StageConfig:
    """peyk-dcr's image is always DEFAULT_DCR_IMAGE — never configurable via yaml, since there's
    no model choice to derive it from at all (see that constant's comment)."""
    stage = _stage(raw)
    if stage.image is None:
        stage.image = DEFAULT_DCR_IMAGE
    return stage


def _ocr_stage(raw: dict) -> StageConfig:
    stage = _stage_with_image_or_vlm(raw, OCR_MODEL_IMAGES)
    if stage.image is None:
        raise ValueError(
            f"ocr.model/cell_ocr.model {stage.backend!r} is not a recognized ocr model "
            f"({sorted(OCR_MODEL_IMAGES)}) or a peyk-vlm model key."
        )
    if stage.backend == "paddleocr-vl" and stage.server_url is None:
        stage.server_url = DEFAULT_VLLM_SERVER_URL
    if stage.backend == "surya" and stage.server_url is None:
        stage.server_url = DEFAULT_SURYA_SERVER_URL
    return stage


def _cell_ocr_backend(config: PipelineConfig) -> str | None:
    """The model that actually ends up processing table-cell crops — config.cell_ocr's if set,
    else whatever config.ocr resolves to (the pre-cell_ocr behavior)."""
    return (config.cell_ocr or config.ocr).backend


def full_table_backend(config: PipelineConfig) -> str | None:
    """Returns "surya", a peyk-vlm model key, or None. Non-None means tsr AND the model that
    would actually process table cells (cell_ocr if set, else ocr — see _cell_ocr_backend)
    resolve to the SAME model, and that model is capable of a one-call full-table
    structure+text path (predict_full for surya, --role table for peyk-vlm) — so table regions
    get routed there instead of the usual structure-then-per-cell-OCR split (see pipeline.py's
    table-routing logic). Keyed off cell_ocr rather than ocr.model directly: ocr.model governs
    non-table text only (dispatch_ocr_batch uses it unconditionally, regardless of this
    function's result) — whether tables need the full-table escape hatch depends on what would
    otherwise OCR their cells, not on what OCRs the rest of the page. So tsr: surya +
    ocr: paddleocr-vl + cell_ocr: surya routes tables through predict_full (returns "surya")
    while tsr: surya + ocr: surya + cell_ocr: paddleocr (explicit) does NOT (returns None) —
    cell_ocr being explicitly set away from surya means the classical
    structure-then-per-cell-OCR split is exactly what was asked for, even though tsr is surya.
    Deliberately independent of layout.model — layout can be anything; this is only about how
    *this* document's tables get processed once detected."""
    tsr_backend = config.tsr.backend
    cell_backend = _cell_ocr_backend(config)
    if tsr_backend is None or tsr_backend != cell_backend:
        return None
    if tsr_backend == "surya" or is_vlm_model(tsr_backend):
        return tsr_backend
    return None


def _validate_tsr_and_cell_ocr(config: PipelineConfig) -> None:
    """Two related hard constraints, both about single-image VLM-style recognizers (Surya's
    RecognitionPredictor/TableRecPredictor, or any peyk-vlm model) never running on isolated
    table-cell crops except as part of a genuine full-table call:

    1. tsr.model being a peyk-vlm model is ONLY valid when cell_ocr also resolves to that exact
       model (full_table_backend matches) — a peyk-vlm model has NO structure-only recognition
       mode at all (confirmed: every peyk-vlm "table" role call does structure+text together,
       see its prompts.py), so tsr.model: <a peyk-vlm model> alone (i.e. meant to pair with a
       separately-configured cell_ocr) is not just unreliable, it's not actually invokable —
       peyk-vlm's CLI has no bare "give me structure only" mode to call. Surya is exempt from
       this specific check since it genuinely supports both (a real structure-only tsr stage,
       and predict_full) — this is why the check is is_vlm_model, not "surya or vlm".
    2. cell_ocr resolving to a VLM-style model (Surya OR a peyk-vlm model) is only reliable when
       tsr resolves to that same model (full-table path, real table-wide context) —
       implementation_plan.md Task 1.5/1.8's isolated-cell-crop finding, reproduced across
       multiple independent VLMs.

    ocr.model never factors into either check (it only governs non-table text); layout.model
    doesn't either."""
    tsr_backend = config.tsr.backend
    if is_vlm_model(tsr_backend) and full_table_backend(config) != tsr_backend:
        raise ValueError(
            f"tsr.model {tsr_backend!r} is a peyk-vlm model, which has no structure-only "
            "recognition mode (only combined structure+text) — it can only be used together "
            f"with cell_ocr.model also set to {tsr_backend!r} (full-table recognition), never "
            "alone. Either set tsr.model to a classical model or 'surya' (which genuinely "
            f"supports structure-only), or set cell_ocr.model to {tsr_backend!r} too."
        )
    cell_backend = _cell_ocr_backend(config)
    is_vlm_style_cell = cell_backend == "surya" or is_vlm_model(cell_backend)
    if is_vlm_style_cell and full_table_backend(config) != cell_backend:
        raise ValueError(
            f"cell_ocr resolves to {cell_backend!r} but tsr.model isn't also {cell_backend!r} "
            "— isolated table-cell crops sent to a single-image VLM recognizer produce "
            "unreliable results (see implementation_plan.md Task 1.5/1.8). Either set "
            "cell_ocr.model to a classical model (e.g. paddleocr), or set tsr.model to "
            f"{cell_backend!r} too (which routes tables through the full-table path instead of "
            "per-cell OCR)."
        )


def load_config(path: Path) -> PipelineConfig:
    raw = yaml.safe_load(path.read_text())
    cell_ocr_raw = raw.get("cell_ocr")
    config = PipelineConfig(
        layout=_layout_stage(raw["layout"]),
        ocr=_ocr_stage(raw["ocr"]),
        dcr=_dcr_stage(raw.get("dcr", {})),
        tsr=_tsr_stage(raw["tsr"]),
        figures=_figures_stage(raw["figures"]),
        cell_ocr=_ocr_stage(cell_ocr_raw) if cell_ocr_raw else None,
        born_digital_min_chars=raw.get("born_digital", {}).get("min_chars_per_page", 20),
        force_scanned=raw.get("born_digital", {}).get("force_scanned", False),
        fullpage=_fullpage_stage(raw.get("fullpage")),
    )
    _validate_tsr_and_cell_ocr(config)
    return config
