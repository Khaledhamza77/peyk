"""vlm-stage cloud credentials — mirrors containers/peyk/run_local.sh's own handling: a Bedrock
bearer token passed as an env var, and a GCP service-account key mounted read-only with
GOOGLE_APPLICATION_CREDENTIALS pointed at it. Neither is validated here (matches the container's
own lazy behavior — pipeline.py._validate_vlm_credentials only errors at dispatch time if a
config actually selects a backend that needs the missing one)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

GCP_KEY_CONTAINER_PATH = "/secrets/gcp-key.json"


@dataclass
class Credentials:
    bedrock_bearer_token: str | None = None
    gcp_key_path: Path | str | None = None

    def env_vars(self) -> dict[str, str]:
        env = {}
        if self.bedrock_bearer_token:
            env["AWS_BEARER_TOKEN_BEDROCK"] = self.bedrock_bearer_token
        if self.gcp_key_path:
            env["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_KEY_CONTAINER_PATH
        return env

    def gcp_mount(self) -> tuple[str, str] | None:
        """(host path, container path) for the GCP key bind mount, or None if not set."""
        if not self.gcp_key_path:
            return None
        return str(Path(self.gcp_key_path).resolve()), GCP_KEY_CONTAINER_PATH
