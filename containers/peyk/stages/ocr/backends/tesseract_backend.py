from pathlib import Path

from .base import OCRBackend, OCRResult

_LANG_CODES = {
    "arabic": "ara",
    "latin": "eng",
}


class TesseractBackend(OCRBackend):
    """Tesseract, via pytesseract. CPU-only — mainline Tesseract has no GPU acceleration path,
    unlike the other backends in this container. Requires the tesseract-ocr apt package plus
    the tesseract-ocr-ara language pack (installed in the Dockerfile)."""

    name = "tesseract"

    def __init__(self, device: str = "gpu", lang: str = "arabic", **_ignored):
        self.lang = _LANG_CODES[lang]

    def load(self) -> None:
        pass

    def predict(self, image_path: Path) -> OCRResult:
        import pytesseract
        from pytesseract import Output

        # Crops are plain PNGs with no DPI metadata, and Tesseract's models are trained
        # assuming ~300 DPI text — without an explicit hint it guesses, which visibly
        # degrades recognition. 300 here must match peyk-orchestrator's/peyk-layout's
        # RENDER_SCALE (300/72). --psm 6 ("assume a single uniform block of text") replaces
        # the default full-page layout analysis (columns/blocks/reading order), which is the
        # wrong tool for an already-isolated region crop and was scrambling word order.
        tess_config = "--dpi 300 --psm 6"
        data = pytesseract.image_to_data(
            str(image_path), lang=self.lang, config=tess_config, output_type=Output.DICT
        )

        words = []
        confidences = []
        for word, conf in zip(data["text"], data["conf"]):
            word = word.strip()
            if not word:
                continue
            words.append(word)
            if float(conf) >= 0:
                confidences.append(float(conf))

        if not words:
            return OCRResult(text="", score=0.0)

        text = " ".join(words)
        score = (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0
        return OCRResult(text=text, score=score)
