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
        self._generation_config = None

    def load(self) -> None:
        from google import genai
        from google.genai import types

        self._client = genai.Client(vertexai=True, project=self.project, location=self.location)

        # Gemini 2.5+ defaults to extended "thinking" (visible as a real thoughtSignature/
        # thoughtsTokenCount even on trivial prompts, confirmed via a live curl test) — real
        # added latency (seconds to tens of seconds) for zero benefit on a straight
        # transcription/description task with no multi-step reasoning involved. thinking_budget=0
        # disables it outright (confirmed via google.genai.types.ThinkingConfig's own field
        # docs: "0 is DISABLED"). Some models may not honor a zero budget (e.g. enforce a
        # minimum) — harmless to request either way.
        #
        # thinking_budget only exists on ThinkingConfig from google-genai>=1.10 — older SDK
        # versions reject it as an unrecognized field (pydantic "Extra inputs are not
        # permitted"). Which SDK version is actually installed depends on how the rest of this
        # image's shared Python environment resolves (surya-ocr and docling-ibm-models pin
        # mutually incompatible ranges of a transitive dependency that also bounds which
        # google-genai versions are installable — see containers/peyk/requirements.txt's own
        # comments), so this environment's SDK version isn't something this code can assume.
        # Computed once in load(), not per predict() call: it can't change mid-process, and this
        # keeps predict() itself simple. Falls back to no thinking-config override (i.e. the
        # SDK's own default — extended thinking stays on, the latency cost documented above,
        # not a correctness problem) rather than failing every single call outright.
        try:
            self._generation_config = types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=0))
        except Exception:
            self._generation_config = None

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
            config=self._generation_config,
        )
        return VLMResult(text=response.text, score=1.0)
