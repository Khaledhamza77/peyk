"""Runs the merged peyk:dev worker+orchestrator image for one job — the Python/docker-py
equivalent of containers/peyk/run_local.sh. Same mounts, same named volumes, same GPU/network
wiring; the only real difference is there's no shell here to fight Windows/MSYS path-rewriting,
since docker-py talks to the Docker Engine API directly.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import docker
from docker.types import DeviceRequest

from .credentials import Credentials
from .sidecars import PEYK_NETWORK, ensure_network

CONTAINER_NAME = "peyk-run"
WORKDIR_VOLUME = "peyk-hotstorage-workdir"
PADDLEX_CACHE_VOLUME = "peyk-paddlex-cache"
CONFIG_CONTAINER_DIR = "/app/stages/orchestrator/config"

_GPU_DEVICE_REQUEST = DeviceRequest(count=-1, capabilities=[["gpu"]])


@dataclass
class RunResult:
    exit_code: int
    output_dir: Path
    logs: str


class PeykRunner:
    def __init__(
        self,
        docker_client: "docker.DockerClient | None" = None,
        image: str = "peyk:dev",
        network: str = PEYK_NETWORK,
    ):
        self.client = docker_client or docker.from_env()
        self.image = image
        self.network = network

    def run(
        self,
        config_path: str | Path,
        input_dir: str | Path,
        output_dir: str | Path,
        credentials: Credentials | None = None,
        extra_args: list[str] | None = None,
        stream_logs: bool = False,
        on_log: Callable[[str], None] | None = None,
    ) -> RunResult:
        """stream_logs=True prints (or calls on_log with) each log chunk as the container
        produces it — useful here specifically because a real job's stdout is exactly the
        "[peyk-orchestrator] ..."/"[peyk-vlm] ..." progress lines seen in run_local.sh's own
        terminal output (layout timing, per-image dispatch), not just a final result. The
        workdir mirror step run_local.sh does at its own end is separate and host-side, not
        part of this container's log — see mirror_workdir_to_host() below. Default is False
        (block silently, return the full log at the end) to keep the common case's return value
        simple; RunResult.logs is populated either way."""
        config_path = Path(config_path).resolve()
        input_dir = Path(input_dir).resolve()
        output_dir = Path(output_dir).resolve()
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        credentials = credentials or Credentials()

        # A config needing zero sidecars (e.g. fullpage with a pure peyk-vlm model) never calls
        # SidecarManager.start(), so nothing else guarantees this network exists yet — attaching
        # to a missing network fails outright (docker.errors.NotFound), Docker does not
        # auto-create it. Idempotent/cheap when a sidecar already created it.
        ensure_network(self.client, self.network)

        try:
            self.client.containers.get(CONTAINER_NAME).remove(force=True)
        except docker.errors.NotFound:
            pass

        volumes = {
            str(config_path.parent): {"bind": CONFIG_CONTAINER_DIR, "mode": "ro"},
            str(input_dir): {"bind": "/hotstorage/input", "mode": "rw"},
            str(output_dir): {"bind": "/hotstorage/output", "mode": "rw"},
            WORKDIR_VOLUME: {"bind": "/hotstorage/workdir", "mode": "rw"},
            PADDLEX_CACHE_VOLUME: {"bind": "/root/.paddlex", "mode": "rw"},
        }
        gcp_mount = credentials.gcp_mount()
        if gcp_mount is not None:
            host_path, container_path = gcp_mount
            volumes[host_path] = {"bind": container_path, "mode": "ro"}

        command = [
            "--config", f"{CONFIG_CONTAINER_DIR}/{config_path.name}",
            "--input", "/hotstorage/input",
            "--output", "/hotstorage/output",
            "--workdir", "/hotstorage/workdir",
            *(extra_args or []),
        ]

        container = self.client.containers.run(
            self.image,
            command=command,
            name=CONTAINER_NAME,
            network=self.network,
            environment=credentials.env_vars(),
            volumes=volumes,
            device_requests=[_GPU_DEVICE_REQUEST],
            detach=True,
        )
        try:
            if stream_logs:
                emit = on_log or (lambda line: print(line, end="", flush=True))
                chunks: list[str] = []
                # follow=True blocks for new output until the container stops logging (i.e.
                # exits) — safe to iterate to exhaustion before wait() below, since by then the
                # container is already done and wait() just returns its exit code immediately.
                for chunk in container.logs(stream=True, follow=True):
                    line = chunk.decode("utf-8", errors="replace")
                    emit(line)
                    chunks.append(line)
                logs = "".join(chunks)
                result = container.wait()
            else:
                result = container.wait()
                logs = container.logs().decode("utf-8", errors="replace")
            exit_code = result.get("StatusCode", 1) if isinstance(result, dict) else int(result)
        finally:
            container.remove(force=True)

        return RunResult(exit_code=exit_code, output_dir=output_dir, logs=logs)

    def mirror_workdir_to_host(self, dest: str | Path) -> None:
        """Copies the peyk-hotstorage-workdir named volume onto a host directory — the same
        trick run_local.sh uses at its own end, as an explicit opt-in rather than automatic:
        this can be a slow copy for a run with many intermediate crop files, so the caller
        decides whether they want it this time."""
        dest = Path(dest).resolve()
        dest.mkdir(parents=True, exist_ok=True)
        self.client.containers.run(
            "alpine",
            command=["cp", "-r", "/w/.", "/out/"],
            volumes={
                WORKDIR_VOLUME: {"bind": "/w", "mode": "ro"},
                str(dest): {"bind": "/out", "mode": "rw"},
            },
            remove=True,
        )
