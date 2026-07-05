from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Region:
    """A detected layout region on a page."""

    page: int
    label: str  # one of: text, table, figure (backend-specific labels get mapped to these)
    score: float
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1 in pixel coords

    def to_dict(self) -> dict:
        return {
            "page": self.page,
            "label": self.label,
            "score": self.score,
            "bbox": list(self.bbox),
        }


class LayoutBackend(ABC):
    """Common interface every layout backend must implement."""

    name: str

    @abstractmethod
    def load(self) -> None:
        """Load model weights. Called once before predict()."""

    @abstractmethod
    def predict(self, image_path: Path) -> list[Region]:
        """Run layout detection on a single page image, return detected regions."""
