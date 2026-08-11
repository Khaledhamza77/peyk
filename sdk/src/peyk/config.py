"""Pipeline config builder — a Python-friendly mirror of
containers/peyk/stages/orchestrator/config.py's schema. PipelineConfig.to_yaml() produces a file
that stage's own load_config() can read directly; the model-name sets and the two tsr/cell_ocr
constraints below are transcribed from that module so a bad config fails fast here in Python
instead of only inside the container.

KNOWN_VLM_MODELS is a static copy of example.yaml's own tier-1/2/3 comment listing (the real
registry lives in the container's stages/vlm package and isn't importable without its runtime
deps) — it can drift behind the container's own list. Treat validate() as a convenience
pre-check, not a guarantee: the container re-validates on load regardless. Pass
allowed_vlm_models to PipelineConfig.validate() (or edit KNOWN_VLM_MODELS directly) if the
container's registry has moved ahead of this copy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .exceptions import ConfigValidationError

LAYOUT_MODELS = frozenset({"pp-doclayout-v2", "doclayout-yolo", "heron", "surya"})
TSR_MODELS = frozenset({"tatr", "rapidtable", "pp-structure-general", "pp-structure-wiring", "tableformer", "surya"})
OCR_MODELS = frozenset({"paddleocr-vl", "paddleocr", "easyocr", "rapidocr", "tesseract", "surya"})

# Transcribed from containers/peyk/stages/vlm/backends/registry.py's MODEL_REGISTRY keys
# directly (not example.yaml's own "peyk-vlm supported models" comment block, which is itself a
# second, independently-drifting copy of the same data) — still a static snapshot, still able to
# drift, per this module's own docstring.
KNOWN_VLM_MODELS = frozenset({
    "claude-haiku", "claude-sonnet-4", "claude-sonnet-4-5", "claude-sonnet-4-6", "claude-sonnet-5",
    "claude-opus-4-5", "claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8", "claude-fable-5",
    "nova-lite", "nova-pro", "nova-2-lite",
    "pixtral-large",
    "kimi-k2-5",
    "gemini-2-5-flash", "gemini-2-5-flash-lite", "gemini-2-5-pro",
    "gemini-3-flash", "gemini-3-1-flash-lite", "gemini-3-1-pro", "gemini-3-5-flash",
    "deepseek-ocr",
})

DEFAULT_VLLM_SERVER_URL = "http://peyk-vllm-paddleocr:8118/v1"
DEFAULT_SURYA_SERVER_URL = "http://peyk-vllm-surya:8000/v1"


@dataclass
class StageConfig:
    model: str | None = None
    lang: str | None = None
    server_url: str | None = None

    def to_dict(self) -> dict:
        raw: dict = {}
        if self.model is not None:
            raw["model"] = self.model
        if self.lang is not None:
            raw["lang"] = self.lang
        if self.server_url is not None:
            raw["server_url"] = self.server_url
        return raw

    @classmethod
    def from_dict(cls, raw: dict) -> "StageConfig":
        return cls(model=raw.get("model"), lang=raw.get("lang"), server_url=raw.get("server_url"))


@dataclass
class SmartSplitConfig:
    enabled: bool = True
    sharpness_threshold_laplacian_var: float = 450
    pixel_safety_margin: float = 0.97
    max_upscale_cap: float = 2.0
    min_scale: float = 1.0

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "sharpness_threshold_laplacian_var": self.sharpness_threshold_laplacian_var,
            "pixel_safety_margin": self.pixel_safety_margin,
            "max_upscale_cap": self.max_upscale_cap,
            "min_scale": self.min_scale,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "SmartSplitConfig":
        defaults = cls()
        return cls(
            enabled=raw.get("enabled", defaults.enabled),
            sharpness_threshold_laplacian_var=raw.get(
                "sharpness_threshold_laplacian_var", defaults.sharpness_threshold_laplacian_var
            ),
            pixel_safety_margin=raw.get("pixel_safety_margin", defaults.pixel_safety_margin),
            max_upscale_cap=raw.get("max_upscale_cap", defaults.max_upscale_cap),
            min_scale=raw.get("min_scale", defaults.min_scale),
        )


def is_vlm_model(model: str | None, allowed_vlm_models: frozenset[str] = KNOWN_VLM_MODELS) -> bool:
    return model is not None and model in allowed_vlm_models


# layout/tsr/ocr/figures all default to an empty StageConfig() (model=None) — valid on their
# own only when `fullpage` is set: the container's load_config() (config.py) checks
# config.fullpage first and, if set, parses these sections with the bare, unvalidated _stage()
# instead of _layout_stage/_tsr_stage/_ocr_stage/_figures_stage, since run.py's main() returns
# via _run_fullpage before any of them are ever dispatched. If you're using the normal per-region
# path instead, set all of layout/tsr/ocr to real choices — validate() (below) will reject
# model=None for them unless fullpage is set.
@dataclass
class PipelineConfig:
    layout: StageConfig = field(default_factory=StageConfig)
    tsr: StageConfig = field(default_factory=StageConfig)
    ocr: StageConfig = field(default_factory=StageConfig)
    figures: StageConfig = field(default_factory=StageConfig)
    dcr: StageConfig = field(default_factory=StageConfig)
    cell_ocr: StageConfig | None = None
    surya_smart_table_split: SmartSplitConfig = field(default_factory=SmartSplitConfig)
    born_digital_min_chars: int = 20
    force_scanned: bool = False
    fullpage: StageConfig | None = None

    def _cell_ocr_backend(self) -> str | None:
        return (self.cell_ocr or self.ocr).model

    def full_table_backend(self, allowed_vlm_models: frozenset[str] = KNOWN_VLM_MODELS) -> str | None:
        tsr_backend = self.tsr.model
        cell_backend = self._cell_ocr_backend()
        if tsr_backend is None or tsr_backend != cell_backend:
            return None
        if tsr_backend == "surya" or is_vlm_model(tsr_backend, allowed_vlm_models):
            return tsr_backend
        return None

    def validate(self, allowed_vlm_models: frozenset[str] = KNOWN_VLM_MODELS) -> None:
        """Raises ConfigValidationError on the first violated constraint. Mirrors
        config.py's load_config(): when `fullpage` is set, layout/tsr/ocr/figures/cell_ocr are
        never dispatched (run.py's main() returns via _run_fullpage first) so load_config skips
        their real validation entirely in that branch — this does the same, checking only
        fullpage itself. Otherwise mirrors _layout_stage/_tsr_stage/_ocr_stage/_figures_stage/
        _fullpage_stage/_validate_tsr_and_cell_ocr."""
        if self.fullpage is not None:
            if self.fullpage.model != "surya" and not is_vlm_model(self.fullpage.model, allowed_vlm_models):
                raise ConfigValidationError(
                    f"fullpage.model {self.fullpage.model!r} is not 'surya' or a recognized "
                    f"peyk-vlm model ({sorted(allowed_vlm_models)})."
                )
            if self.fullpage.lang is not None:
                raise ConfigValidationError(
                    "fullpage.lang is a no-op — neither the surya nor the vlm fullpage dispatch "
                    "(run.py's _run_fullpage) reads --lang, and that path never calls "
                    "normalize_digits() the way the normal per-region path's assemble_document() "
                    "does for ocr.lang. Leave fullpage.lang unset."
                )
            return

        if self.layout.model not in LAYOUT_MODELS:
            raise ConfigValidationError(
                f"layout.model {self.layout.model!r} is not a recognized layout model "
                f"({sorted(LAYOUT_MODELS)}) — layout can never be done by a peyk-vlm model."
            )
        if self.tsr.model not in TSR_MODELS and not is_vlm_model(self.tsr.model, allowed_vlm_models):
            raise ConfigValidationError(
                f"tsr.model {self.tsr.model!r} is not a recognized tsr model "
                f"({sorted(TSR_MODELS)}) or a peyk-vlm model key."
            )
        for stage, label in ((self.ocr, "ocr"), (self.cell_ocr, "cell_ocr")):
            if stage is None:
                continue
            if stage.model not in OCR_MODELS and not is_vlm_model(stage.model, allowed_vlm_models):
                raise ConfigValidationError(
                    f"{label}.model {stage.model!r} is not a recognized ocr model "
                    f"({sorted(OCR_MODELS)}) or a peyk-vlm model key."
                )
        if self.figures.model is not None and not is_vlm_model(self.figures.model, allowed_vlm_models):
            raise ConfigValidationError(
                f"figures.model {self.figures.model!r} is not a recognized peyk-vlm model "
                f"({sorted(allowed_vlm_models)})."
            )
        self._validate_tsr_and_cell_ocr(allowed_vlm_models)

    def _validate_tsr_and_cell_ocr(self, allowed_vlm_models: frozenset[str]) -> None:
        tsr_backend = self.tsr.model
        if is_vlm_model(tsr_backend, allowed_vlm_models) and self.full_table_backend(allowed_vlm_models) != tsr_backend:
            raise ConfigValidationError(
                f"tsr.model {tsr_backend!r} is a peyk-vlm model, which has no structure-only "
                "recognition mode (only combined structure+text) — it can only be used together "
                f"with cell_ocr.model also set to {tsr_backend!r} (full-table recognition), never "
                "alone. Either set tsr.model to a classical model or 'surya' (which genuinely "
                f"supports structure-only), or set cell_ocr.model to {tsr_backend!r} too."
            )
        cell_backend = self._cell_ocr_backend()
        is_vlm_style_cell = cell_backend == "surya" or is_vlm_model(cell_backend, allowed_vlm_models)
        if is_vlm_style_cell and self.full_table_backend(allowed_vlm_models) != cell_backend:
            raise ConfigValidationError(
                f"cell_ocr resolves to {cell_backend!r} but tsr.model isn't also {cell_backend!r} "
                "— isolated table-cell crops sent to a single-image VLM recognizer produce "
                "unreliable results. Either set cell_ocr.model to a classical model (e.g. "
                f"paddleocr), or set tsr.model to {cell_backend!r} too (which routes tables "
                "through the full-table path instead of per-cell OCR)."
            )

    def sidecar_requirements(self, allowed_vlm_models: frozenset[str] = KNOWN_VLM_MODELS) -> set[str]:
        """Which of {"surya", "paddleocr"} sidecars this config actually needs, based on every
        stage's resolved backend — used to avoid starting a sidecar nothing in the config will
        ever call."""
        # layout.model included too: "surya" is a valid, independent layout choice (LAYOUT_MODELS
        # includes it, and pipeline.py's run_layout() dispatches to the surya sidecar for it) —
        # missing this meant a layout-only surya config passed validate() but never got its
        # sidecar started, so run_layout() would hit a connection refused against a sidecar that
        # was never running.
        backends = {self.layout.model, self.tsr.model, self.ocr.model}
        if self.cell_ocr is not None:
            backends.add(self.cell_ocr.model)
        if self.fullpage is not None:
            backends.add(self.fullpage.model)
        needed: set[str] = set()
        if "surya" in backends:
            needed.add("surya")
        if "paddleocr-vl" in backends:
            needed.add("paddleocr")
        return needed

    def to_dict(self) -> dict:
        if self.fullpage is not None:
            # run.py's main() returns via _run_fullpage before layout/tsr/ocr/figures/cell_ocr/
            # surya_smart_table_split/born_digital are ever read — every one of them is specific
            # to the normal per-region path (prepare_document/dispatch_documents), which fullpage
            # bypasses entirely. Writing them out here would just be confusing, unused detail
            # sitting next to the one section that actually matters.
            return {"fullpage": self.fullpage.to_dict()}

        raw: dict = {
            "layout": self.layout.to_dict(),
            "tsr": self.tsr.to_dict(),
            "ocr": self.ocr.to_dict(),
            "figures": self.figures.to_dict(),
            "surya_smart_table_split": self.surya_smart_table_split.to_dict(),
            "born_digital": {
                "min_chars_per_page": self.born_digital_min_chars,
                "force_scanned": self.force_scanned,
            },
        }
        if self.dcr.to_dict():
            raw["dcr"] = self.dcr.to_dict()
        if self.cell_ocr is not None:
            raw["cell_ocr"] = self.cell_ocr.to_dict()
        return raw

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @classmethod
    def from_dict(cls, raw: dict) -> "PipelineConfig":
        # .get(key, {}) rather than raw[key]: a fullpage config's YAML omits layout/tsr/ocr/
        # figures entirely (see to_dict()) — missing means "unused placeholder", same as the
        # container's own load_config() fullpage branch. validate() (not this method) is what
        # actually enforces they're real choices for the normal per-region path.
        cell_ocr_raw = raw.get("cell_ocr")
        fullpage_raw = raw.get("fullpage")
        return cls(
            layout=StageConfig.from_dict(raw.get("layout", {})),
            tsr=StageConfig.from_dict(raw.get("tsr", {})),
            ocr=StageConfig.from_dict(raw.get("ocr", {})),
            figures=StageConfig.from_dict(raw.get("figures", {})),
            dcr=StageConfig.from_dict(raw.get("dcr", {})),
            cell_ocr=StageConfig.from_dict(cell_ocr_raw) if cell_ocr_raw else None,
            surya_smart_table_split=SmartSplitConfig.from_dict(raw.get("surya_smart_table_split", {})),
            born_digital_min_chars=raw.get("born_digital", {}).get("min_chars_per_page", 20),
            force_scanned=raw.get("born_digital", {}).get("force_scanned", False),
            fullpage=StageConfig.from_dict(fullpage_raw) if fullpage_raw else None,
        )

    @classmethod
    def from_yaml(cls, text: str) -> "PipelineConfig":
        return cls.from_dict(yaml.safe_load(text))

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> "PipelineConfig":
        return cls.from_yaml(Path(path).read_text())
