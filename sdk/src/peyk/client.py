"""Peyk — the facade tying config + credentials + sidecars + the main container run together.
This is what most users of the package should import and use directly; the lower-level pieces
(PipelineConfig, SidecarManager, PeykRunner, Credentials) stay available for anyone who wants
finer control.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import docker

from .config import PipelineConfig
from .credentials import Credentials
from .runner import PeykRunner, RunResult
from .sidecars import PEYK_NETWORK, SidecarManager


class Peyk:
    def __init__(
        self,
        image: str = "peyk:dev",
        network: str = PEYK_NETWORK,
        docker_client: "docker.DockerClient | None" = None,
    ):
        self.client = docker_client or docker.from_env()
        self.sidecars = SidecarManager(self.client, network=network)
        self.runner = PeykRunner(self.client, image=image, network=network)
        self.credentials = Credentials()
        self._config: PipelineConfig | None = None
        self._config_path: Path | None = None

    def build_image(self, context_dir: str | Path, tag: str | None = None) -> None:
        """Convenience wrapper for building peyk:dev from a local checkout of this repo
        (containers/peyk) — not required if the image already exists locally or was pulled
        from wherever it's published."""
        self.client.images.build(path=str(Path(context_dir).resolve()), tag=tag or self.runner.image)

    def configure(self, config: PipelineConfig, config_dir: str | Path, filename: str = "config.yaml") -> Path:
        """Validates and writes `config` as YAML into config_dir, ready for `run()` to mount.
        Returns the written file's path."""
        config.validate()
        config_dir = Path(config_dir).resolve()
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / filename
        config_path.write_text(config.to_yaml())
        self._config = config
        self._config_path = config_path
        return config_path

    def set_credentials(
        self,
        bedrock_bearer_token: str | None = None,
        gcp_key_path: str | Path | None = None,
    ) -> None:
        self.credentials = Credentials(bedrock_bearer_token=bedrock_bearer_token, gcp_key_path=gcp_key_path)

    def ensure_sidecars(self, wait: bool = True, **sidecar_overrides) -> set[str]:
        """Starts (and, if wait=True, waits ready for) whichever sidecars the configured
        pipeline actually needs, per PipelineConfig.sidecar_requirements(). Must be called after
        configure(). sidecar_overrides is forwarded to SidecarManager.start (surya tuning knobs
        only)."""
        if self._config is None:
            raise RuntimeError("call configure() before ensure_sidecars()")
        needed = self._config.sidecar_requirements()
        for name in needed:
            overrides = sidecar_overrides if name == "surya" else {}
            self.sidecars.start(name, **overrides)
        if wait:
            for name in needed:
                self.sidecars.wait_ready(name)
        return needed

    def stop_sidecars(self) -> None:
        self.sidecars.stop_all()

    def run(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        extra_args: list[str] | None = None,
        stream_logs: bool = False,
        on_log: Callable[[str], None] | None = None,
    ) -> RunResult:
        if self._config_path is None:
            raise RuntimeError("call configure() before run()")
        return self.runner.run(
            config_path=self._config_path,
            input_dir=input_dir,
            output_dir=output_dir,
            credentials=self.credentials,
            extra_args=extra_args,
            stream_logs=stream_logs,
            on_log=on_log,
        )

    def mirror_workdir_to_host(self, dest: str | Path) -> None:
        self.runner.mirror_workdir_to_host(dest)
