from .base import OCRBackend, OCRResult
from .paddleocr import PaddleOCRBackend
from .easyocr_backend import EasyOCRBackend
from .rapidocr_backend import RapidOCRBackend
from .tesseract_backend import TesseractBackend

BACKENDS = {
    "paddleocr": PaddleOCRBackend,
    "easyocr": EasyOCRBackend,
    "rapidocr": RapidOCRBackend,
    "tesseract": TesseractBackend,
}

__all__ = ["OCRBackend", "OCRResult", "BACKENDS"]
