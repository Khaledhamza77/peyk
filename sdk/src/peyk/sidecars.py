"""Lifecycle management for the two persistent vLLM sidecars, transcribed from
containers/peyk-vllm-surya/start.sh and containers/peyk-vllm-paddleocr/start.sh. Talks to Docker
via docker-py rather than shelling out to `docker run` — no MSYS/Windows path-rewriting concerns
here, since no shell sits between this code and the Docker Engine API.

One deliberate deviation from peyk-vllm-paddleocr/start.sh: that script never publishes a host
port (peyk-orchestrator's old sibling-container dispatch only ever needed peyk-net), but this SDK
runs readiness checks from the host, so its sidecar definition below also publishes 8118 to the
host. Harmless — the container is still reachable from peyk-net by name exactly as before.
"""
from __future__ import annotations

import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Literal

import docker
from docker.types import DeviceRequest

from .exceptions import SidecarNotReadyError

SidecarName = Literal["surya", "paddleocr"]

PEYK_NETWORK = "peyk-net"

_GPU_DEVICE_REQUEST = DeviceRequest(count=-1, capabilities=[["gpu"]])


def ensure_network(client: "docker.DockerClient", network: str) -> None:
    """Idempotent `docker network create` — a container.run() with `network=<name>` fails
    outright (docker.errors.NotFound: "network ... not found") rather than auto-creating it, so
    ANY caller attaching a container to `network` needs to have called this first. Shared by
    SidecarManager (below) and PeykRunner.run(): a config needing zero sidecars (e.g. fullpage
    with a pure peyk-vlm model) never calls SidecarManager.start(), so PeykRunner.run() cannot
    rely on that path having created the network already — it must ensure it itself too."""
    try:
        client.networks.get(network)
    except docker.errors.NotFound:
        client.networks.create(network)


@dataclass(frozen=True)
class SidecarSpec:
    container_name: str
    image: str
    port: int  # both host and container side
    ready_path: str
    # Overridable tuning env vars / command flags, keyed the same way start.sh reads them.
    command: tuple[str, ...] = ()
    environment: dict[str, str] = field(default_factory=dict)
    volumes: dict[str, dict[str, str]] = field(default_factory=dict)
    extra_kwargs: dict = field(default_factory=dict)


def _surya_spec(
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 18000,
    max_num_seqs: int = 8,
    enforce_eager: bool = False,
) -> SidecarSpec:
    command = [
        "--model", "datalab-to/surya-ocr-2",
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
        "--max-num-seqs", str(max_num_seqs),
        "--mm-processor-kwargs", '{"min_pixels": 3136, "max_pixels": 6291456}',
    ]
    if enforce_eager:
        command.append("--enforce-eager")
    return SidecarSpec(
        container_name="peyk-vllm-surya",
        image="vllm/vllm-openai:v0.20.1",
        port=8119,
        ready_path="/v1/models",
        command=tuple(command),
        volumes={
            "peyk-vllm-surya-cache": {"bind": "/root/.cache/huggingface", "mode": "rw"},
            "peyk-vllm-surya-torch-cache": {"bind": "/root/.cache/vllm", "mode": "rw"},
        },
        extra_kwargs={"ipc_mode": "host", "ports": {"8000/tcp": 8119}},
    )


def _paddleocr_spec() -> SidecarSpec:
    return SidecarSpec(
        container_name="peyk-vllm-paddleocr",
        image="ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu",
        port=8118,
        ready_path="/v1/models",
        command=(
            "paddleocr", "genai_server", "--model_name", "PaddleOCR-VL-0.9B",
            "--host", "0.0.0.0", "--port", "8118", "--backend", "vllm",
            "--backend_config", "/tmp/vllm_config.yml",
        ),
        environment={"PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True"},
        volumes={"peyk-vllm-paddleocr-cache": {"bind": "/home/paddleocr/.paddlex", "mode": "rw"}},
        extra_kwargs={"ports": {"8118/tcp": 8118}},
    )


# Cold starts are real and documented (surya: ~14 minutes on a 12GB card per that script's own
# tuning notes) — default wait_ready timeouts reflect that instead of a short default that trips
# over the sidecar's own normal startup cost.
DEFAULT_READY_TIMEOUT_S = {"surya": 20 * 60, "paddleocr": 10 * 60}


class SidecarManager:
    def __init__(self, docker_client: "docker.DockerClient | None" = None, network: str = PEYK_NETWORK):
        self.client = docker_client or docker.from_env()
        self.network = network

    def ensure_network(self) -> None:
        ensure_network(self.client, self.network)

    def _spec(self, name: SidecarName, **overrides) -> SidecarSpec:
        if name == "surya":
            return _surya_spec(**overrides)
        if name == "paddleocr":
            if overrides:
                raise ValueError("paddleocr sidecar has no tunable overrides")
            return _paddleocr_spec()
        raise ValueError(f"unknown sidecar {name!r} (expected 'surya' or 'paddleocr')")

    def _fixup_paddleocr_volume_ownership(self, spec: SidecarSpec) -> None:
        """A fresh named volume is created root-owned; peyk-vllm-paddleocr's image runs as a
        non-root `paddleocr` user, which then can't write into it ("PermissionError:
        /home/paddleocr/.paddlex/temp") — see that container's own start.sh, which fixes this
        with a one-off `--user root ... --entrypoint chown` run before the real server starts.
        Surya's image runs as root by default (that script's own comment), so it alone needs no
        equivalent. Harmless/fast on subsequent runs once ownership is already correct."""
        self.client.containers.run(
            spec.image,
            command=["-R", "paddleocr:paddleocr", "/home/paddleocr/.paddlex"],
            entrypoint="chown",
            user="root",
            volumes=spec.volumes,
            remove=True,
        )

    def start(self, name: SidecarName, **overrides) -> None:
        """Removes any pre-existing container of the same name, then starts a fresh one
        (matching start.sh's own `docker rm -f ... || true` before `docker run`)."""
        self.ensure_network()
        spec = self._spec(name, **overrides)
        try:
            self.client.containers.get(spec.container_name).remove(force=True)
        except docker.errors.NotFound:
            pass
        if name == "paddleocr":
            self._fixup_paddleocr_volume_ownership(spec)
        self.client.containers.run(
            spec.image,
            command=list(spec.command),
            name=spec.container_name,
            network=self.network,
            environment=spec.environment,
            volumes=spec.volumes,
            device_requests=[_GPU_DEVICE_REQUEST],
            detach=True,
            **spec.extra_kwargs,
        )

    def is_ready(self, name: SidecarName) -> bool:
        spec = self._spec(name)
        url = f"http://localhost:{spec.port}{spec.ready_path}"
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def wait_ready(self, name: SidecarName, timeout_s: float | None = None, poll_interval_s: float = 5.0) -> None:
        timeout_s = DEFAULT_READY_TIMEOUT_S[name] if timeout_s is None else timeout_s
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.is_ready(name):
                return
            time.sleep(poll_interval_s)
        raise SidecarNotReadyError(
            f"{name} sidecar didn't answer {self._spec(name).ready_path} within {timeout_s:.0f}s"
        )

    def stop(self, name: SidecarName) -> None:
        spec = self._spec(name)
        try:
            self.client.containers.get(spec.container_name).remove(force=True)
        except docker.errors.NotFound:
            pass

    def stop_all(self) -> None:
        for name in ("surya", "paddleocr"):
            self.stop(name)
