from pathlib import Path

from .base import OCRBackend, OCRResult

# Matches the prompt PaddleX's own "PaddleOCR-VL" pipeline uses for OCR-classified regions
# (paddlex/inference/pipelines/paddleocr_vl/pipeline.py).
_OCR_QUERY = "OCR:"

# A single text-region crop should never legitimately need more than a couple hundred
# tokens; this also bounds the damage of the runaway-repetition failure mode below.
_MAX_NEW_TOKENS = 256

# PaddleX's local (in-process) PaddleOCR-VL predictor silently drops repetition_penalty —
# its vendored `PaddleOCRVLForConditionalGeneration.generate()` override hardcodes its
# kwargs to just {max_new_tokens, use_cache}, so there was no way to guard against the
# observed failure mode (greedy decoding occasionally falling into a repeat loop and
# grinding to the token cap, e.g. "2.00 ₰ ₰ ₰ ₰..."). Routed through a vllm-server backend
# instead, PaddleX's DocVLMGenAIClientPredictor forwards this straight into the request's
# extra_body, where vLLM actually applies it. 1.15 is a starting point: high enough to break
# short repeat loops, low enough to not suppress legitimate repeated characters/digits in
# names or table cells.
_REPETITION_PENALTY = 1.15


class PaddleOCRVLBackend(OCRBackend):
    """PaddleOCR-VL-0.9B (VLM: NaViT-style vision encoder + ERNIE-4.5-0.3B decoder), served by
    a persistent vLLM server (see peyk-vllm-paddleocr/) rather than loaded in-process via
    PaddleX's local `create_model()` path. The local path was tried first and abandoned: its
    naive single-token-at-a-time generate loop syncs GPU->CPU every step (near-idle GPU,
    single CPU thread pegged), its batch size is hardcoded to 1 with no override, and it
    offers no way to pass repetition_penalty/similar decode guards. All three are structural
    to that code path, not fixable via config — see docs/build_notes.md. vLLM's own decode
    loop (continuous batching, CUDA graphs) avoids the first two, and its OpenAI-compatible
    server exposes decode-guard params the local path couldn't. This backend is a thin HTTP
    client: PaddleX's own `DocVLMGenAIClientPredictor` (selected via `engine_config` below)
    speaks the OpenAI chat-completions protocol to it, so no bespoke client code is needed
    here."""

    name = "paddleocr-vl"

    def __init__(self, device: str = "gpu", lang: str = "arabic", server_url: str | None = None, **_ignored):
        self.server_url = server_url
        self._model = None

    def load(self) -> None:
        from paddlex import create_model

        self._model = create_model(
            model_name="PaddleOCR-VL-0.9B",
            engine="genai_client",
            engine_config={"backend": "vllm-server", "server_url": self.server_url},
        )

    def predict(self, image_path: Path) -> OCRResult:
        if self._model is None:
            raise RuntimeError("Backend not loaded; call load() first")

        for result in self._model.predict(
            {"image": str(image_path), "query": _OCR_QUERY},
            max_new_tokens=_MAX_NEW_TOKENS,
            repetition_penalty=_REPETITION_PENALTY,
        ):
            text = result["result"]
            # A generative VLM has no per-token recognition confidence like PaddleOCR's CTC
            # decoder does; report 1.0 for a successful generation for a uniform OCRResult shape.
            return OCRResult(text=text, score=1.0)

        return OCRResult(text="", score=0.0)
