from pathlib import Path

from prompts import PROMPTS

from .base import VLMBackend, VLMResult

# Matches the rest of this project (build_notes.md Task 0.4: Claude Haiku 4.5's confirmed
# working region, and the cross-region inference profile ID it must be invoked through).
_DEFAULT_REGION = "eu-north-1"

_IMAGE_FORMATS = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".webp": "webp", ".gif": "gif"}


class BedrockVLMBackend(VLMBackend):
    """Any Bedrock model reachable via the `converse()` API (Claude family, Nova, ...) —
    invocation mechanics only, the specific model is `model_id` (see backends/registry.py).
    Built/verified first: Claude Haiku 4.5 on Bedrock is already confirmed zero-friction
    working in this project (Task 0.4), unlike either Vertex path."""

    name = "bedrock"

    def __init__(self, model_id: str, region: str = _DEFAULT_REGION, **_ignored):
        self.model_id = model_id
        self.region = region
        self._client = None

    def load(self) -> None:
        import boto3

        # No explicit credential handling here — boto3/botocore resolves auth automatically,
        # and bedrock-runtime supports two schemes (confirmed via its service model:
        # signingName "bedrock", auth ["aws.auth#sigv4", "smithy.api#httpBearerAuth"]):
        # (1) a Bedrock API key via the AWS_BEARER_TOKEN_BEDROCK env var (botocore derives this
        # name itself from the signing name) — preferred for this container: scoped to just
        # Bedrock, no need to mount a full IAM user's credential file; or (2) the normal SigV4
        # credential chain (~/.aws profile, IAM role, etc.) if that env var isn't set. Verified
        # working with just `--env-file <file containing AWS_BEARER_TOKEN_BEDROCK=...>` and no
        # ~/.aws mount at all — see implementation_plan.md Task 1.6 and this container's README.
        self._client = boto3.client("bedrock-runtime", region_name=self.region)

    def predict(self, image_path: Path, role: str) -> VLMResult:
        if self._client is None:
            raise RuntimeError("Backend not loaded; call load() first")

        image_format = _IMAGE_FORMATS.get(image_path.suffix.lower())
        if image_format is None:
            raise ValueError(f"Unsupported image extension for Bedrock converse(): {image_path.suffix}")

        response = self._client.converse(
            modelId=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"image": {"format": image_format, "source": {"bytes": image_path.read_bytes()}}},
                        {"text": PROMPTS[role]},
                    ],
                }
            ],
        )
        text = response["output"]["message"]["content"][0]["text"]
        return VLMResult(text=text, score=1.0)
