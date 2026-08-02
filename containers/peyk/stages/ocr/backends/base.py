from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OCRResult:
    """Recognized text for a single text-region crop."""

    text: str
    score: float

    def to_dict(self) -> dict:
        return {"text": self.text, "score": self.score}


class OCRBackend(ABC):
    """Common interface every OCR backend must implement."""

    name: str

    @abstractmethod
    def load(self) -> None:
        """Load model weights. Called once before predict()."""

    @abstractmethod
    def predict(self, image_path: Path) -> OCRResult:
        """Run text recognition on a single text-region crop, return recognized text."""
