from pathlib import Path

from .base import OCRBackend, OCRResult


def _lang_rec_config():
    # Deferred import so this module doesn't require `rapidocr` at import time (mirrors the
    # other backends' pattern of importing heavy deps inside load()).
    from rapidocr.utils.typings import LangRec, ModelType, OCRVersion

    # RapidOCR's PaddlePaddle-engine Arabic recognition model only exists under PP-OCRv5 (not
    # the default PP-OCRv6 — verified empirically: PP-OCRv6's lang_type allowlist has no
    # Arabic entry at all, see rapidocr/utils/model_resolver.py PP_OCRV6_LANGS). Values must be
    # actual Enum members, not their string equivalents — RapidOCR's param parser rejects
    # plain strings for these three keys (TypeError: "must be Enum Type").
    return {
        "arabic": {
            "lang_type": LangRec.ARABIC,
            "ocr_version": OCRVersion.PPOCRV5,
            "model_type": ModelType.MOBILE,
        },
        "latin": {
            "lang_type": LangRec.EN,
            "ocr_version": OCRVersion.PPOCRV6,
            "model_type": ModelType.SMALL,
        },
    }


class RapidOCRBackend(OCRBackend):
    """RapidOCR, configured to use the PaddlePaddle inference engine (reuses the already-
    installed paddlepaddle-gpu) rather than onnxruntime-gpu, avoiding a second CUDA-matched
    wheel/pin. RTL reordering (Arabic) is handled internally by RapidOCR itself
    (ch_ppocr_rec/main.py RTL_LANGS), unlike PaddleX's pipeline which needed python-bidi."""

    name = "rapidocr"

    def __init__(self, device: str = "gpu", lang: str = "arabic"):
        self.use_cuda = device == "gpu"
        self.lang = lang
        self._engine = None

    def load(self) -> None:
        from rapidocr import RapidOCR
        from rapidocr.utils.typings import EngineType

        self.rec_config = _lang_rec_config()[self.lang]
        self._engine = RapidOCR(
            params={
                "Det.engine_type": EngineType.PADDLE,
                "Cls.engine_type": EngineType.PADDLE,
                "Rec.engine_type": EngineType.PADDLE,
                "Rec.ocr_version": self.rec_config["ocr_version"],
                "Rec.model_type": self.rec_config["model_type"],
                "Rec.lang_type": self.rec_config["lang_type"],
                "EngineConfig.paddle.use_cuda": self.use_cuda,
            }
        )

    def predict(self, image_path: Path) -> OCRResult:
        if self._engine is None:
            raise RuntimeError("Backend not loaded; call load() first")

        result = self._engine(str(image_path))
        if not result.txts:
            return OCRResult(text="", score=0.0)

        text = "\n".join(result.txts)
        score = sum(result.scores) / len(result.scores)
        return OCRResult(text=text, score=float(score))
