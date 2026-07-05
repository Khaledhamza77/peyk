from .base import OCRBackend, OCRResult
from .paddleocr_vl import PaddleOCRVLBackend

BACKENDS = {
    "paddleocr-vl": PaddleOCRVLBackend,
}

__all__ = ["OCRBackend", "OCRResult", "BACKENDS"]
