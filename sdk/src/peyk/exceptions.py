class PeykError(Exception):
    """Base class for all errors raised by this package."""


class ConfigValidationError(PeykError):
    """A PipelineConfig violates one of the orchestrator's own config constraints."""


class SidecarNotReadyError(PeykError):
    """A vLLM sidecar didn't answer its /v1/models endpoint within the given timeout."""


class MissingCredentialsError(PeykError):
    """A selected backend needs a credential (Bedrock token, GCP key) that wasn't provided."""


class NotConfiguredError(PeykError):
    """A Peyk method that needs configure() to have run first (ensure_sidecars(), run()) was
    called before it did."""
