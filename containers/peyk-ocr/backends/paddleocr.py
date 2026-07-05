from pathlib import Path

from .base import OCRBackend, OCRResult


_LANG_REC_MODELS = {
    "arabic": "arabic_PP-OCRv5_mobile_rec",
    "latin": "PP-OCRv5_server_rec",
}


class PaddleOCRBackend(OCRBackend):
    """Full PaddleOCR det+rec pipeline (PaddleX's "OCR" pipeline). Layout regions handed to
    this container can span multiple lines, and PP-OCRv5's recognition model only accepts a
    single line at a fixed height — so text detection runs first to split the crop into lines,
    then each line is recognized and joined back together. Company documents are predominantly
    Arabic, so that's the default rec model; pass lang="latin" for English/Latin-script crops."""

    name = "paddleocr"

    def __init__(self, device: str = "gpu", lang: str = "arabic"):
        self.device = device
        self.rec_model_name = _LANG_REC_MODELS[lang]
        self._pipeline = None

    def load(self) -> None:
        from paddlex.inference.pipelines import create_pipeline, load_pipeline_config

        config = load_pipeline_config("OCR")
        config["SubModules"]["TextRecognition"]["model_name"] = self.rec_model_name
        self._pipeline = create_pipeline(config=config, device=self.device)

    def predict(self, image_path: Path) -> OCRResult:
        if self._pipeline is None:
            raise RuntimeError("Backend not loaded; call load() first")

        for result in self._pipeline.predict(str(image_path)):
            texts = result["rec_texts"]
            scores = result["rec_scores"]
            if not texts:
                return OCRResult(text="", score=0.0)
            text = "\n".join(texts)
            score = sum(scores) / len(scores)
            return OCRResult(text=text, score=float(score))

        return OCRResult(text="", score=0.0)
