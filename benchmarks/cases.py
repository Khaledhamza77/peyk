"""The default sweep: stage-isolated, not a cross-product.

layout(4) x tsr(5) x ocr(6) x cell_ocr x figures is thousands of runs, and almost none of them
answer a question anyone asked. Instead each group below freezes every slot but one and sweeps
that slot, so a difference in the resulting numbers is attributable to the one thing that changed.
That is roughly 20 runs rather than thousands, and it is the only shape in which per-stage latency
figures mean anything.

Two constraints from the orchestrator's own config validation are respected here, and cases that
would violate them are simply not generated (see containers/peyk/stages/orchestrator/config.py and
config/example.yaml):
  - a VLM model on `tsr` is only valid when the SAME model is on `cell_ocr` (one combined
    structure+text call);
  - `surya` on `cell_ocr` is only valid when `tsr` is also `surya` (the predict_full path) --
    isolated per-cell crops sent to a VLM recognise badly, which this project established
    empirically on two independent VLMs.

`fullpage` cases are included deliberately as the control arm: they bypass layout/TSR/OCR
entirely, one model call per page. They are what tells you whether the per-region pipeline's
extra complexity is buying anything at all.
"""
from __future__ import annotations

from peyk import PipelineConfig, StageConfig

from .harness import BenchmarkCase

# The reference pipeline every single-slot sweep varies from. Chosen to be entirely self-hosted so
# that a swapped-in managed model shows up as a clean delta against a no-network baseline, rather
# than one network-bound stage being compared against another.
_BASE_LANG = "arabic"


def _base() -> PipelineConfig:
    config = PipelineConfig(
        layout=StageConfig(model="heron"),
        tsr=StageConfig(model="tableformer"),
        ocr=StageConfig(model="paddleocr", lang=_BASE_LANG),
        cell_ocr=StageConfig(model="paddleocr", lang=_BASE_LANG),
        figures=StageConfig(model="gemini-3-1-flash-lite"),
    )
    # force_scanned is ON for every sweep except the ablation that exists to vary it.
    #
    # This is load-bearing, not a stylistic default. Most of this repo's sample PDFs are
    # born-digital, and with force_scanned off the orchestrator routes their text through the DCR
    # text-layer path -- no OCR dispatch at all. An OCR sweep run that way measures nothing:
    # confirmed empirically, a run over cib_sample.pdf produced 0 region crops and no `ocr` stage
    # event whatsoever, so all six backends would have tied at zero seconds.
    #
    # It also matches the config this project actually runs (hotstorage/peyk-config/config.yaml
    # sets force_scanned: true), so the swept numbers describe the real deployment path.
    config.force_scanned = True
    # No benchmark case, harness, or report reads a single _viz.png/_aug_viz.png -- so the real
    # per-crop draw+encode cost of producing them is pure overhead here, paid on every run of
    # every case for artifacts nobody looks at. Off across the whole sweep.
    config.visualize = False
    return config


def _with(**overrides) -> PipelineConfig:
    config = _base()
    for slot, value in overrides.items():
        setattr(config, slot, value)
    return config


def layout_sweep() -> list[BenchmarkCase]:
    """All four layout backends against an otherwise identical pipeline. Layout runs once per
    page, so the denominator here is seconds per page."""
    return [
        BenchmarkCase(f"layout={model}", _with(layout=StageConfig(model=model)),
                      note="layout backend swept; all other slots fixed")
        for model in ("heron", "doclayout-yolo", "pp-doclayout-v2", "surya")
    ]


def ocr_sweep() -> list[BenchmarkCase]:
    """Whole-region text recognition only -- cell_ocr is pinned to paddleocr throughout so table
    cells do not move between backends at the same time and confound the comparison.

    tesseract is the interesting row for compute cost specifically: it is CPU-only, so it should
    show ~zero VRAM delta while every other backend here reserves GPU memory."""
    cases = [
        BenchmarkCase(f"ocr={model}", _with(ocr=StageConfig(model=model, lang=_BASE_LANG)),
                      note="self-hosted OCR backend swept; cell_ocr pinned to paddleocr")
        for model in ("paddleocr", "easyocr", "rapidocr", "tesseract", "paddleocr-vl")
    ]
    cases.append(
        BenchmarkCase(
            "ocr=gemini-3-1-flash-lite",
            _with(ocr=StageConfig(model="gemini-3-1-flash-lite", lang=_BASE_LANG)),
            note="managed API -- network-bound, no local VRAM cost, per-call billing",
        )
    )
    return cases


def tsr_sweep() -> list[BenchmarkCase]:
    """Self-hosted TSR backends, plus the two paired modes the validator allows."""
    cases = [
        BenchmarkCase(f"tsr={model}", _with(tsr=StageConfig(model=model)),
                      note="self-hosted TSR backend swept; cell_ocr pinned to paddleocr")
        for model in ("tableformer", "tatr", "rapidtable", "pp-structure-general")
    ]
    # surya on tsr requires surya on cell_ocr -- this is the predict_full whole-table path, a
    # structurally different mode rather than just another TSR backend, so it is labelled as one.
    cases.append(
        BenchmarkCase(
            "tsr=surya + cell_ocr=surya (predict_full)",
            _with(tsr=StageConfig(model="surya"), cell_ocr=StageConfig(model="surya")),
            note="whole-table recognition in one call, no per-cell OCR dispatch",
        )
    )
    # A VLM on tsr is likewise only valid paired with itself on cell_ocr.
    cases.append(
        BenchmarkCase(
            "tsr=claude-sonnet-4-5 + cell_ocr=same",
            _with(
                tsr=StageConfig(model="claude-sonnet-4-5"),
                cell_ocr=StageConfig(model="claude-sonnet-4-5", lang=_BASE_LANG),
            ),
            note="managed VLM doing structure+text in one combined call",
        )
    )
    return cases


def fullpage_sweep() -> list[BenchmarkCase]:
    """The control arm: no layout, no TSR, no per-region OCR -- one model call per page.

    If a fullpage case is both faster and qualitatively acceptable on the demo documents, that is
    a genuine finding about the per-region pipeline, and it should be presented as one rather than
    quietly omitted."""
    cases = []
    for model in ("surya", "gemini-3-1-flash-lite", "gemini-3-5-flash"):
        config = _base()
        config.fullpage = StageConfig(model=model)
        cases.append(
            BenchmarkCase(f"fullpage={model}", config,
                          note="bypasses layout/TSR/OCR entirely -- one call per page")
        )
    return cases


def smart_split_ablation() -> list[BenchmarkCase]:
    """On/off for the Surya blurred-table split correction. This is the one knob in the config
    whose *quality* effect this project has genuinely characterised (7 tables, 3 documents -- see
    docs-personal/surya/improvement.md), so pairing that with its measured latency cost is a
    defensible slide: the accuracy claim is this project's own recorded observation, at a stated
    sample size, and the cost number is measured here."""
    cases = []
    for enabled in (False, True):
        config = _base()
        config.tsr = StageConfig(model="surya")
        config.cell_ocr = StageConfig(model="surya")
        config.surya_smart_table_split.enabled = enabled
        cases.append(
            BenchmarkCase(
                f"smart_table_split={'on' if enabled else 'off'}",
                config,
                note="split/upscale correction for blurred table crops",
            )
        )
    return cases


def born_digital_ablation() -> list[BenchmarkCase]:
    """force_scanned on/off. Worth its own slide: the live config in hotstorage/peyk-config
    currently sets force_scanned: true, which routes born-digital pages through OCR/VLM even
    though the DCR path would extract their text layer directly -- no model, no GPU, no API
    call. This measures exactly what that setting costs."""
    cases = []
    for forced in (True, False):
        config = _base()
        # Flattened onto PipelineConfig by the SDK (born_digital_min_chars / force_scanned); it
        # only becomes a nested `born_digital:` block on the way out, in to_dict(). This is the
        # one sweep that overrides _base()'s force_scanned=True, since varying it IS the ablation.
        config.force_scanned = forced
        cases.append(
            BenchmarkCase(
                f"force_scanned={'on' if forced else 'off'}",
                config,
                note="off = born-digital pages use the DCR text-layer path (no model at all)",
            )
        )
    return cases


SWEEPS = {
    "layout": layout_sweep,
    "ocr": ocr_sweep,
    "tsr": tsr_sweep,
    "fullpage": fullpage_sweep,
    "smart-split": smart_split_ablation,
    "born-digital": born_digital_ablation,
}


def build(names: list[str]) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for name in names:
        if name not in SWEEPS:
            raise KeyError(f"unknown sweep {name!r}; available: {', '.join(sorted(SWEEPS))}")
        cases.extend(SWEEPS[name]())
    return cases
