import pytest

from peyk import ConfigValidationError, PipelineConfig, StageConfig


def _base_config(**overrides) -> PipelineConfig:
    defaults = dict(
        layout=StageConfig(model="heron"),
        tsr=StageConfig(model="surya"),
        ocr=StageConfig(model="gemini-3-1-flash-lite", lang="arabic"),
        figures=StageConfig(model="gemini-3-1-flash-lite"),
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def test_valid_config_round_trips_through_yaml():
    config = _base_config(cell_ocr=StageConfig(model="surya"))
    config.validate()

    reloaded = PipelineConfig.from_yaml(config.to_yaml())

    assert reloaded.layout.model == "heron"
    assert reloaded.tsr.model == "surya"
    assert reloaded.ocr.model == "gemini-3-1-flash-lite"
    assert reloaded.ocr.lang == "arabic"
    assert reloaded.cell_ocr.model == "surya"
    assert reloaded.figures.model == "gemini-3-1-flash-lite"


def test_matches_example_yaml_structure(tmp_path):
    example = (
        "layout:\n  model: heron\n"
        "tsr:\n  model: surya\n"
        "ocr:\n  model: gemini-3-1-flash-lite\n  lang: arabic\n"
        "cell_ocr:\n  model: surya\n"
        "figures:\n  model: gemini-3-1-flash-lite\n"
        "born_digital:\n  min_chars_per_page: 20\n  force_scanned: true\n"
    )
    config = PipelineConfig.from_yaml(example)
    config.validate()
    assert config.force_scanned is True
    assert config.born_digital_min_chars == 20


def test_rejects_unknown_layout_model():
    config = _base_config(layout=StageConfig(model="not-a-model"))
    with pytest.raises(ConfigValidationError, match="layout.model"):
        config.validate()


def test_vlm_tsr_requires_matching_cell_ocr():
    # tsr set to a peyk-vlm model with no matching cell_ocr -> invalid (no structure-only mode).
    # ocr deliberately set to a DIFFERENT model here, since cell_ocr falls back to ocr.model
    # when unset — using the same model for both would accidentally satisfy the constraint.
    config = _base_config(tsr=StageConfig(model="gemini-3-1-flash-lite"), ocr=StageConfig(model="tesseract"))
    with pytest.raises(ConfigValidationError, match="has no structure-only"):
        config.validate()

    # Same tsr model, now paired with a matching cell_ocr -> valid.
    config = _base_config(
        tsr=StageConfig(model="gemini-3-1-flash-lite"),
        cell_ocr=StageConfig(model="gemini-3-1-flash-lite"),
    )
    config.validate()


def test_vlm_style_cell_ocr_requires_matching_tsr():
    # cell_ocr resolves to surya (VLM-style) but tsr does not match -> invalid.
    config = _base_config(tsr=StageConfig(model="tableformer"), cell_ocr=StageConfig(model="surya"))
    with pytest.raises(ConfigValidationError, match="isolated table-cell crops"):
        config.validate()

    # Matching tsr -> valid.
    config = _base_config(tsr=StageConfig(model="surya"), cell_ocr=StageConfig(model="surya"))
    config.validate()


def test_fullpage_only_construction_needs_no_other_stage_args():
    # layout/tsr/ocr/figures all fall back to their own defaults (model=None) — fullpage is
    # configurable on its own, the same way cell_ocr or dcr are, with no separate factory
    # method needed.
    config = PipelineConfig(fullpage=StageConfig(model="gemini-3-1-flash-lite"))
    config.validate()

    assert config.fullpage.model == "gemini-3-1-flash-lite"
    # The written YAML contains only what fullpage actually uses — layout/tsr/ocr/figures/
    # cell_ocr/surya_smart_table_split/born_digital are all specific to the normal per-region
    # path (which fullpage bypasses entirely, see run.py's _run_fullpage) and would just be
    # confusing, unused detail if written out.
    assert config.to_dict() == {"fullpage": {"model": "gemini-3-1-flash-lite"}}
    # A vlm-backed fullpage model needs no vLLM sidecar of its own.
    assert config.sidecar_requirements() == set()

    reloaded = PipelineConfig.from_yaml(config.to_yaml())
    assert reloaded.fullpage.model == "gemini-3-1-flash-lite"
    assert reloaded.layout.model is None
    assert reloaded.tsr.model is None
    assert reloaded.ocr.model is None


def test_fullpage_only_with_surya_needs_surya_sidecar():
    config = PipelineConfig(fullpage=StageConfig(model="surya"))
    config.validate()
    assert config.sidecar_requirements() == {"surya"}


def test_fullpage_rejects_lang_as_a_no_op():
    # Neither run.py's surya nor vlm fullpage dispatch reads --lang, and that path never calls
    # normalize_digits() the way ocr.lang does in the normal per-region path — set it and
    # validate() should say so rather than silently accepting a field that has no effect.
    config = PipelineConfig(fullpage=StageConfig(model="surya", lang="arabic"))
    with pytest.raises(ConfigValidationError, match="fullpage.lang is a no-op"):
        config.validate()


def test_fullpage_rejects_unrecognized_model():
    config = PipelineConfig(fullpage=StageConfig(model="not-a-model"))
    with pytest.raises(ConfigValidationError, match="fullpage.model"):
        config.validate()


def test_sidecar_requirements_only_lists_what_is_used():
    surya_only = _base_config(tsr=StageConfig(model="surya"), cell_ocr=StageConfig(model="surya"))
    assert surya_only.sidecar_requirements() == {"surya"}

    paddleocr_only = _base_config(
        tsr=StageConfig(model="tableformer"),
        ocr=StageConfig(model="paddleocr-vl"),
    )
    assert paddleocr_only.sidecar_requirements() == {"paddleocr"}

    neither = _base_config(tsr=StageConfig(model="tableformer"), ocr=StageConfig(model="tesseract"))
    assert neither.sidecar_requirements() == set()
