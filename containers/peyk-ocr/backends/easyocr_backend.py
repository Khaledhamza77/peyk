from pathlib import Path

from .base import OCRBackend, OCRResult

_LANG_LISTS = {
    "arabic": ["ar", "en"],
    "latin": ["en"],
}


class EasyOCRBackend(OCRBackend):
    """EasyOCR (CRAFT detector + CRNN recognizer), PyTorch-based. GPU support is free here —
    reuses the torch/torchvision already pinned in the Dockerfile for Granite-Docling, no new
    dependency or CUDA pin needed; just Reader(gpu=True) (the default)."""

    name = "easyocr"

    def __init__(self, device: str = "gpu", lang: str = "arabic"):
        self.gpu = device == "gpu"
        self.lang_list = _LANG_LISTS[lang]
        self._reader = None

    def load(self) -> None:
        import easyocr

        self._reader = easyocr.Reader(self.lang_list, gpu=self.gpu)

    def predict(self, image_path: Path) -> OCRResult:
        if self._reader is None:
            raise RuntimeError("Backend not loaded; call load() first")

        results = self._reader.readtext(str(image_path))
        if not results:
            return OCRResult(text="", score=0.0)

        texts = [r[1] for r in results]
        scores = [r[2] for r in results]
        return OCRResult(text="\n".join(texts), score=float(sum(scores) / len(scores)))
