import base64
from pathlib import Path

from prompts import PROMPTS

from .base import VLMBackend, VLMResult

_DEFAULT_PROJECT = "peyk-501209"

_MIME_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


class VertexMaaSBackend(VLMBackend):
    """Vertex AI Model-Garden-as-a-Service partner models reachable via Vertex's
    OpenAI-compatible chat-completions endpoint (deepseek-ai/deepseek-ocr-maas@001,
    mistralai/mistral-ocr-2505@001 — see backends/registry.py) — mirrors the pattern
    peyk-paddleocr-vl already uses against its own vLLM server (openai SDK client, image as a
    base64 data URI in an image_url content part).

    CONFIRMED (implementation_plan.md Task 1.6, live curl investigation): these MaaS partner
    models are only reachable via Vertex's **global** endpoint
    (`aiplatform.googleapis.com`, `locations/global`) — a regional endpoint (e.g.
    `europe-west1-aiplatform.googleapis.com`) 400s with "is only available via global
    endpoint." Unlike Gemini (native, regional), location is not configurable for this
    backend. `deepseek-ai/deepseek-ocr-maas@001` resolved past the model-lookup step at the
    global endpoint (auth/authorization confirmed fine, a real request just hadn't finished
    within a 2-minute bound in that manual test); `mistralai/mistral-ocr-2505@001` still 404s
    even at global — "not found or your project does not have access to it" — likely needs an
    explicit per-model enable/terms-acceptance step in the Model Garden console, not a code
    problem here."""

    name = "vertex-maas"

    _GLOBAL_BASE_URL_TEMPLATE = "https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/endpoints/openapi"

    def __init__(self, model_id: str, project: str = _DEFAULT_PROJECT, **_ignored):
        self.model_id = model_id
        self.project = project
        self._credentials = None

    def load(self) -> None:
        import google.auth

        self._credentials, _ = google.auth.default(scopes=_SCOPES)

    def _client(self):
        # Refreshed on every predict() call (not just once in load()) since a bearer token is
        # short-lived (~1h) and this backend may be used across a long --watch run; refresh()
        # is cheap and a no-op if the current token still has headroom.
        import google.auth.transport.requests
        from openai import OpenAI

        self._credentials.refresh(google.auth.transport.requests.Request())
        base_url = self._GLOBAL_BASE_URL_TEMPLATE.format(project=self.project)
        return OpenAI(base_url=base_url, api_key=self._credentials.token)

    def predict(self, image_path: Path, role: str) -> VLMResult:
        if self._credentials is None:
            raise RuntimeError("Backend not loaded; call load() first")

        mime_type = _MIME_TYPES.get(image_path.suffix.lower())
        if mime_type is None:
            raise ValueError(f"Unsupported image extension for Vertex MaaS chat completions: {image_path.suffix}")

        b64_image = base64.b64encode(image_path.read_bytes()).decode("ascii")
        response = self._client().chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPTS[role]},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}},
                    ],
                }
            ],
        )
        return VLMResult(text=response.choices[0].message.content, score=1.0)
