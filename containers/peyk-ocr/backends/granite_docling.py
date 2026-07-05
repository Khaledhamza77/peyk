from pathlib import Path

from .base import OCRBackend, OCRResult

_OCR_PROMPT = "OCR the full page to markdown."


class GraniteDoclingBackend(OCRBackend):
    """Granite-Docling-258M (IBM, Idefics3 architecture + Granite ~165M decoder), run via
    plain `transformers` (AutoModelForImageTextToText/AutoProcessor) rather than the full
    `docling` package, which pulls in a whole PDF/layout/table pipeline this container
    doesn't need.

    Confirmed gotcha (not yet independently verified against our own hardware, only against
    documented reports): on Turing (compute capability 7.5, e.g. the T4 this is meant to run
    on in the cloud), bf16/fp16 inference silently produces garbage output rather than
    erroring. Ampere+/Ada (the local RTX 3500 Ada, sm_89) runs bf16 correctly. We therefore
    pick dtype from the detected compute capability rather than hardcoding one.

    Known quality issue (see Task 1.3 build notes): output includes DocTags location-tag
    markup (e.g. "<loc_132><loc_45>...") even with the documented plain-OCR prompt, and this
    model's Arabic transcription had real character-level errors on a crop PaddleOCR-VL got
    perfect. Not yet fixed — recorded as-is rather than blocking on it."""

    name = "granite-docling"
    model_id = "ibm-granite/granite-docling-258M"

    def __init__(self, device: str = "gpu", **_ignored):
        self.device = device
        self._model = None
        self._processor = None

    def _select_dtype(self):
        import torch

        if not torch.cuda.is_available():
            return torch.float32
        major, _minor = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float32

    def load(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._dtype = self._select_dtype()
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, dtype=self._dtype
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        self._model.eval()

    def predict(self, image_path: Path) -> OCRResult:
        if self._model is None or self._processor is None:
            raise RuntimeError("Backend not loaded; call load() first")

        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": _OCR_PROMPT},
                ],
            }
        ]
        prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(text=prompt, images=[image], return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            generated = self._model.generate(**inputs, max_new_tokens=1024)
        trimmed = generated[:, inputs["input_ids"].shape[1] :]
        text = self._processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

        # Generative model, no native per-token confidence; same tradeoff as paddleocr-vl.
        return OCRResult(text=text, score=1.0)
