"""Pipeline config: which backend/image each stage uses, or whether it's stubbed."""
from dataclasses import dataclass
from pathlib import Path

import yaml


# Which image each ocr backend runs in — see containers/peyk-simple-ocr/ and
# containers/peyk-paddleocr-vl/ (the split rationale is in implementation_plan.md Task 1.3).
# Derived automatically from `backend` rather than requiring `image` in the yaml too: the
# pairing is fully determined by the backend choice, so making both configurable independently
# just invites a config that sets backend: paddleocr-vl but forgets to also flip image (or
# vice versa) — a mismatch `run_docker_stage` has no way to catch, since it just docker-runs
# whatever image/model string it's given.
OCR_BACKEND_IMAGES = {
    "paddleocr-vl": "peyk-paddleocr-vl:dev",
    "paddleocr": "peyk-simple-ocr:dev",
    "easyocr": "peyk-simple-ocr:dev",
    "rapidocr": "peyk-simple-ocr:dev",
    "tesseract": "peyk-simple-ocr:dev",
}

# peyk-paddleocr-vl's default --server-url already matches this (see that container's
# run.py), so this only matters if peyk-vllm-paddleocr is reachable at a different address.
DEFAULT_VLLM_SERVER_URL = "http://peyk-vllm-paddleocr:8118/v1"


@dataclass
class StageConfig:
    image: str | None = None
    backend: str | None = None
    lang: str = "arabic"
    stub: bool = False
    role: str = "fallback"  # peyk-vlm only: "fallback" or "primary"
    server_url: str | None = None  # ocr only: peyk-paddleocr-vl's vLLM server URL


@dataclass
class PipelineConfig:
    layout: StageConfig
    ocr: StageConfig
    dcr: StageConfig
    tsr: StageConfig
    vlm: StageConfig
    born_digital_min_chars: int = 20
    force_scanned: bool = False


def _stage(raw: dict) -> StageConfig:
    return StageConfig(
        image=raw.get("image"),
        backend=raw.get("backend"),
        lang=raw.get("lang", "arabic"),
        stub=raw.get("stub", False),
        role=raw.get("role", "fallback"),
        server_url=raw.get("server_url"),
    )


def _ocr_stage(raw: dict) -> StageConfig:
    stage = _stage(raw)
    if stage.image is None and stage.backend in OCR_BACKEND_IMAGES:
        stage.image = OCR_BACKEND_IMAGES[stage.backend]
    if stage.backend == "paddleocr-vl" and stage.server_url is None:
        stage.server_url = DEFAULT_VLLM_SERVER_URL
    return stage


def load_config(path: Path) -> PipelineConfig:
    raw = yaml.safe_load(path.read_text())
    return PipelineConfig(
        layout=_stage(raw["layout"]),
        ocr=_ocr_stage(raw["ocr"]),
        dcr=_stage(raw.get("dcr", {"stub": True})),
        tsr=_stage(raw.get("tsr", {"stub": True})),
        vlm=_stage(raw.get("vlm", {"stub": True})),
        born_digital_min_chars=raw.get("born_digital", {}).get("min_chars_per_page", 20),
        force_scanned=raw.get("born_digital", {}).get("force_scanned", False),
    )
