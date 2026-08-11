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
        # Crops handed to this container are already-flat, upright region images rendered
        # directly from a PDF page (peyk-orchestrator's pipeline.py), not photographed
        # scans — so the doc-preprocessor's orientation classification and UVDoc unwarping
        # (built for skewed/curved camera captures) have nothing real to correct and can
        # only distort an already-clean crop. Disabled here, not just left at its default.
        config["use_doc_preprocessor"] = False
        # Default 1.5 expands each detected text box to 1.5x its tight-fit size before
        # handing it to recognition. Raised here since Arabic diacritics (fatha/damma/
        # kasra/shadda/sukun) can sit just outside a box that's only loosened by the
        # default amount, getting clipped before recognition ever sees them.
        config["SubModules"]["TextDetection"]["unclip_ratio"] = 2.0
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
