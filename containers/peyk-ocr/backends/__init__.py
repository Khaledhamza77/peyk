from .base import OCRBackend, OCRResult
from .paddleocr import PaddleOCRBackend
from .paddleocr_vl import PaddleOCRVLBackend
from .granite_docling import GraniteDoclingBackend
from .easyocr_backend import EasyOCRBackend
from .rapidocr_backend import RapidOCRBackend
from .tesseract_backend import TesseractBackend

BACKENDS = {
    "paddleocr": PaddleOCRBackend,
    "paddleocr-vl": PaddleOCRVLBackend,
    "granite-docling": GraniteDoclingBackend,
    "easyocr": EasyOCRBackend,
    "rapidocr": RapidOCRBackend,
    "tesseract": TesseractBackend,
}

__all__ = ["OCRBackend", "OCRResult", "BACKENDS"]
