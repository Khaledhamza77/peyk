from pathlib import Path

from .base import OCRBackend, OCRResult

# Matches the prompt PaddleX's own "PaddleOCR-VL" pipeline uses for OCR-classified regions
# (paddlex/inference/pipelines/paddleocr_vl/pipeline.py).
_OCR_QUERY = "OCR:"


class PaddleOCRVLBackend(OCRBackend):
    """PaddleOCR-VL-0.9B (VLM: NaViT-style vision encoder + ERNIE-4.5-0.3B decoder), run via
    PaddleX's native `doc_vlm` model group rather than the full "PaddleOCR-VL" pipeline — the
    full pipeline runs its own internal layout detection, which would duplicate peyk-layout's
    job. No new ML framework: reuses the already-installed paddlepaddle-gpu, no torch/transformers
    involved. bf16-vs-fp32 device selection is handled internally by PaddleX (is_bfloat16_available),
    so unlike the transformers-based backends this needs no manual T4/Ada dtype branching."""

    name = "paddleocr-vl"

    def __init__(self, device: str = "gpu", **_ignored):
        self.device = device
        self._model = None

    def load(self) -> None:
        from paddlex import create_model

        self._model = create_model(model_name="PaddleOCR-VL-0.9B", device=self.device)

    def predict(self, image_path: Path) -> OCRResult:
        if self._model is None:
            raise RuntimeError("Backend not loaded; call load() first")

        for result in self._model.predict({"image": str(image_path), "query": _OCR_QUERY}):
            text = result["result"]
            # A generative VLM has no per-token recognition confidence like PaddleOCR's CTC
            # decoder does; report 1.0 for a successful generation for a uniform OCRResult shape.
            return OCRResult(text=text, score=1.0)

        return OCRResult(text="", score=0.0)
