from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VLMResult:
    """Model output for a single image, whatever the role. `text` holds plain text for the
    ocr/figure roles, HTML for table (fed to markdownify() downstream), or Markdown for
    fullpage — the shape is uniform so callers don't need a role-specific result type."""

    text: str
    # Hardcoded 1.0 on success: none of Bedrock/Vertex-Gemini/Vertex-MaaS expose a real
    # per-call confidence score, same tradeoff already documented for OCRResult
    # (peyk-paddleocr-vl/peyk-surya).
    score: float

    def to_dict(self) -> dict:
        return {"text": self.text, "score": self.score}


class VLMBackend(ABC):
    """Common interface every provider adapter implements. One instance serves any role via
    predict()'s `role` argument — role selects the prompt (see prompts.py), not a different
    backend instance or class."""

    name: str

    @abstractmethod
    def load(self) -> None:
        """Construct/authenticate the client. Called once before predict()."""

    @abstractmethod
    def predict(self, image_path: Path, role: str) -> VLMResult:
        """Run the model against a single image (a region crop, or a whole rendered page for
        role="fullpage") for the given role ("ocr" | "figure" | "table" | "fullpage")."""
