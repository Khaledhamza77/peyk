from pathlib import Path

from prompts import PROMPTS

from .base import VLMBackend, VLMResult

# Matches the GCP project/region already verified for Gemini in build_notes.md Task 0.4
# (GCP-era) / architecture_proposal.md's multi-cloud decision.
_DEFAULT_PROJECT = "peyk-501209"
_DEFAULT_LOCATION = "europe-west1"

_MIME_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}


class VertexGeminiBackend(VLMBackend):
    """Gemini family via Vertex AI's native generateContent, using the google-genai SDK's
    Vertex backend (`Client(vertexai=True, ...)`) rather than the public Gemini API — keeps
    inference inside the private GCP project boundary per pipeline.md's privacy constraint."""

    name = "vertex-gemini"

    def __init__(self, model_id: str, project: str = _DEFAULT_PROJECT, location: str = _DEFAULT_LOCATION, **_ignored):
        self.model_id = model_id
        self.project = project
        self.location = location
        self._client = None

    def load(self) -> None:
        from google import genai

        self._client = genai.Client(vertexai=True, project=self.project, location=self.location)

    def predict(self, image_path: Path, role: str) -> VLMResult:
        if self._client is None:
            raise RuntimeError("Backend not loaded; call load() first")

        from google.genai import types

        mime_type = _MIME_TYPES.get(image_path.suffix.lower())
        if mime_type is None:
            raise ValueError(f"Unsupported image extension for Gemini generateContent(): {image_path.suffix}")

        response = self._client.models.generate_content(
            model=self.model_id,
            contents=[
                types.Part.from_bytes(data=image_path.read_bytes(), mime_type=mime_type),
                PROMPTS[role],
            ],
            # Gemini 2.5+ defaults to extended "thinking" (visible as a real thoughtSignature/
            # thoughtsTokenCount even on trivial prompts, confirmed via a live curl test) — real
            # added latency (seconds to tens of seconds) for zero benefit on a straight
            # transcription/description task with no multi-step reasoning involved.
            # thinking_budget=0 disables it outright (confirmed via google.genai.types
            # .ThinkingConfig's own field docs: "0 is DISABLED"). Some models may not honor a
            # zero budget (e.g. enforce a minimum) — harmless to request either way.
            config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=0)),
        )
        return VLMResult(text=response.text, score=1.0)
