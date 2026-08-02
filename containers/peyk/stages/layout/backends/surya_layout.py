from pathlib import Path

from .base import LayoutBackend, Region


class SuryaLayoutBackend(LayoutBackend):
    """Not yet implemented. Candidate per pipeline.md; wire up when prioritized."""

    name = "surya-layout"

    def load(self) -> None:
        raise NotImplementedError("surya-layout backend not implemented yet (see docs/pipeline.md)")

    def predict(self, image_path: Path) -> list[Region]:
        raise NotImplementedError("surya-layout backend not implemented yet (see docs/pipeline.md)")
