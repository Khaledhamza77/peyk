"""Cookie-cutter model registry: adding a new invokable model is one entry here, not a new
class. Three provider adapter classes handle invocation mechanics generically over any
model_id + any role; MODEL_REGISTRY only says which provider + which model.

Model keys are plain model names — no "bedrock-"/"vertex-" prefix. Which provider a model
belongs to is looked up here, never inferred from the key's spelling: peyk-orchestrator's
config.py queries this registry directly via `run.py --list-models` (prints
"<key>\\t<provider>") rather than guessing from a naming convention, which is exactly the kind
of assumption that let a typo silently pass validation before this was fixed
(implementation_plan.md Task 1.6/1.7)."""
from .base import VLMBackend
from .bedrock import BedrockVLMBackend
from .vertex_gemini import VertexGeminiBackend
from .vertex_maas import VertexMaaSBackend

MODEL_REGISTRY: dict[str, dict] = {
    # --- Bedrock: every IMAGE-input-capable foundation model confirmed live in eu-north-1 via
    # `aws bedrock list-foundation-models` / `list-inference-profiles` (implementation_plan.md
    # Task 1.6) — not a curated subset, the full set this account can actually invoke. Most
    # entries use the "eu." cross-region inference profile ID, not the bare model ID (required
    # for on-demand invocation — see build_notes.md Task 0.4); a couple of exceptions below.
    #
    # Anthropic models each require their OWN separate agreement + one-time account use-case
    # form before they're invokable (confirmed: Claude Sonnet 5's agreement offerId differs
    # from Haiku 4.5's) — only claude-haiku has actually been accepted/verified so far. Every
    # other Anthropic entry here will fail loudly with a clear Bedrock error
    # (ResourceNotFoundException) until its own agreement is accepted the same way (see
    # implementation_plan.md Task 1.6's CLI commands) — registered anyway, cookie-cutter style,
    # rather than left out, since accepting the gate is an account action, not a code change.
    "claude-haiku": {"provider": "bedrock", "model_id": "eu.anthropic.claude-haiku-4-5-20251001-v1:0"},
    "claude-sonnet-4": {"provider": "bedrock", "model_id": "eu.anthropic.claude-sonnet-4-20250514-v1:0"},
    "claude-sonnet-4-5": {"provider": "bedrock", "model_id": "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"},
    "claude-sonnet-4-6": {"provider": "bedrock", "model_id": "eu.anthropic.claude-sonnet-4-6"},
    "claude-sonnet-5": {"provider": "bedrock", "model_id": "eu.anthropic.claude-sonnet-5"},
    "claude-opus-4-5": {"provider": "bedrock", "model_id": "eu.anthropic.claude-opus-4-5-20251101-v1:0"},
    "claude-opus-4-6": {"provider": "bedrock", "model_id": "eu.anthropic.claude-opus-4-6-v1"},
    "claude-opus-4-7": {"provider": "bedrock", "model_id": "eu.anthropic.claude-opus-4-7"},
    "claude-opus-4-8": {"provider": "bedrock", "model_id": "eu.anthropic.claude-opus-4-8"},
    # Only a "global." inference profile exists for this one (no "eu." variant) — confirmed via
    # list-inference-profiles, not assumed.
    "claude-fable-5": {"provider": "bedrock", "model_id": "global.anthropic.claude-fable-5"},
    # Amazon-native — no agreement gate at all (agreementAvailability: AVAILABLE out of the
    # box, confirmed live), unlike every Anthropic entry above.
    "nova-lite": {"provider": "bedrock", "model_id": "eu.amazon.nova-lite-v1:0"},
    "nova-pro": {"provider": "bedrock", "model_id": "eu.amazon.nova-pro-v1:0"},
    "nova-2-lite": {"provider": "bedrock", "model_id": "eu.amazon.nova-2-lite-v1:0"},
    # Mistral AI, Bedrock-native — no agreement gate either (confirmed live), a real
    # alternative to the currently-404ing Vertex Mistral-OCR MaaS path if Mistral quality is
    # wanted without touching GCP at all.
    "pixtral-large": {"provider": "bedrock", "model_id": "eu.mistral.pixtral-large-2502-v1:0"},
    # ON_DEMAND inference type — invoked via the bare model ID directly, no cross-region
    # inference profile exists or is needed for this one (confirmed via
    # list-foundation-models's inferenceTypesSupported). Untested for OCR/document quality —
    # not in pipeline.md's original candidate list, registered for comparison.
    "kimi-k2-5": {"provider": "bedrock", "model_id": "moonshotai.kimi-k2.5"},
    # --- Vertex AI
    # Gemini family: every text+vision generative model in the Model Garden catalog
    # (`gcloud ai model-garden models list`, `CAN_PREDICT: Yes`) — deliberately excludes
    # image-*generation* variants (flash-image/pro-image, wrong direction: text->image, not
    # image->text), TTS variants, `gemini-embedding-*` (not generative), the computer-use-agent
    # preview, live-audio, and omni-audio variants — none of those fit this container's
    # image-in/text-or-html-out shape.
    "gemini-2-5-flash": {"provider": "vertex-gemini", "model_id": "gemini-2.5-flash"},
    "gemini-2-5-flash-lite": {"provider": "vertex-gemini", "model_id": "gemini-2.5-flash-lite"},
    "gemini-2-5-pro": {"provider": "vertex-gemini", "model_id": "gemini-2.5-pro"},
    # Gemini 3.x: confirmed 404 at the regional endpoint (europe-west1) for every one of these —
    # a live curl comparison (regional vs. "global" vs. bare "global-aiplatform.googleapis.com",
    # the last of which doesn't exist as a host) showed only the true global endpoint
    # (host aiplatform.googleapis.com, `locations/global` in the path) actually resolves these
    # models — same gotcha already hit and fixed for the Vertex MaaS backends. `location`
    # override below is `VertexGeminiBackend`'s existing constructor param, not new code.
    "gemini-3-flash": {"provider": "vertex-gemini", "model_id": "gemini-3-flash-preview", "location": "global"},
    # gemini-3-pro (gemini-3-pro-preview) removed: 404s even at the global endpoint, a separate
    # narrower preview-access restriction (not the endpoint bug the other Gemini 3.x entries
    # hit) — gemini-3-1-pro below already covers this quality tier and works.
    "gemini-3-1-flash-lite": {"provider": "vertex-gemini", "model_id": "gemini-3.1-flash-lite", "location": "global"},
    "gemini-3-1-pro": {"provider": "vertex-gemini", "model_id": "gemini-3.1-pro-preview", "location": "global"},
    "gemini-3-5-flash": {"provider": "vertex-gemini", "model_id": "gemini-3.5-flash", "location": "global"},
    # DeepSeek: deepseek-ocr-maas@001 is the ONLY MaaS-predictable DeepSeek-OCR variant —
    # confirmed via the catalog: deepseek-ocr@deepseek-ocr and deepseek-ocr-2@deepseek-ocr-2 are
    # both CAN_DEPLOY-only (self-deploy GPU endpoint, same exclusion reasoning as Qwen2.5-VL in
    # pipeline.md), not usable via this container's managed-pay-per-call design at all.
    "deepseek-ocr": {"provider": "vertex-maas", "model_id": "deepseek-ai/deepseek-ocr-maas@001"},
    # mistral-ocr removed (implementation_plan.md Task 1.6): blocked on a publisher-wide Model
    # Garden enablement gate for this project (confirmed via testing mistral-medium-3/
    # mistral-small-2503 too, both 404 identically), not a code problem — registered anyway,
    # cookie-cutter style, rather than left permanently broken.
}

PROVIDER_CLASSES: dict[str, type[VLMBackend]] = {
    "bedrock": BedrockVLMBackend,
    "vertex-gemini": VertexGeminiBackend,
    "vertex-maas": VertexMaaSBackend,
}


def get_backend(model_key: str) -> VLMBackend:
    entry = dict(MODEL_REGISTRY[model_key])
    provider = entry.pop("provider")
    return PROVIDER_CLASSES[provider](**entry)
