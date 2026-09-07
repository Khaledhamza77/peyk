"""Runs the merged peyk:dev worker+orchestrator image for one job — the Python/docker-py
equivalent of containers/peyk/run_local.sh. Same mounts, same named volumes, same GPU/network
wiring; the only real difference is there's no shell here to fight Windows/MSYS path-rewriting,
since docker-py talks to the Docker Engine API directly.
"""
from __future__ import annotations

import shlex
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import docker

from .credentials import Credentials
from .history import ArtifactStore, JobStore, relative_to_workdir
from .sidecars import GPU_DEVICE_REQUEST, PEYK_NETWORK, ensure_network

CONTAINER_NAME = "peyk-run"
WORKDIR_VOLUME = "peyk-hotstorage-workdir"
PADDLEX_CACHE_VOLUME = "peyk-paddlex-cache"
CONFIG_CONTAINER_DIR = "/app/stages/orchestrator/config"


@dataclass
class RunResult:
    exit_code: int
    output_dir: Path
    logs: str
    job_id: str


class PeykRunner:
    def __init__(
        self,
        docker_client: "docker.DockerClient | None" = None,
        image: str = "peyk:dev",
        network: str = PEYK_NETWORK,
        job_store: JobStore | None = None,
        artifact_store: ArtifactStore | None = None,
    ):
        self.client = docker_client or docker.from_env()
        self.image = image
        self.network = network
        self.job_store = job_store or JobStore()
        self.artifact_store = artifact_store or ArtifactStore()

    def run(
        self,
        config_path: str | Path,
        input_dir: str | Path,
        output_dir: str | Path,
        credentials: Credentials | None = None,
        extra_args: list[str] | None = None,
        stream_logs: bool = False,
        on_log: Callable[[str], None] | None = None,
        persist_artifacts: bool = False,
    ) -> RunResult:
        """stream_logs=True prints (or calls on_log with) each log chunk as the container
        produces it — useful here specifically because a real job's stdout is exactly the
        "[peyk-orchestrator] ..."/"[peyk-vlm] ..." progress lines seen in run_local.sh's own
        terminal output (layout timing, per-image dispatch), not just a final result. The
        workdir mirror step run_local.sh does at its own end is separate and host-side, not
        part of this container's log — see mirror_workdir_to_host() below. Default is False
        (block silently, return the full log at the end) to keep the common case's return value
        simple; RunResult.logs is populated either way.

        Every call is recorded in self.job_store regardless of any other option here — a job row
        up front, one events row per @@PEYK-EVENT@@ line the container emits (see
        containers/peyk/stages/orchestrator/events.py), ingested from the log this method already
        reads. This part is cheap (text only) and always on.

        persist_artifacts=True additionally copies each dispatched stage's own input/output
        directory (crops, per-region model JSON/HTML, viz PNGs) out of the shared
        peyk-hotstorage-workdir volume into self.artifact_store, stage-partitioned by job — see
        history.py's own docstring for the layout. Off by default: a single job's table-cell OCR
        crops alone can be hundreds of files, so this is opt-in per call, not automatic."""
        config_path = Path(config_path).resolve()
        input_dir = Path(input_dir).resolve()
        output_dir = Path(output_dir).resolve()
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        credentials = credentials or Credentials()
        job_id = uuid.uuid4().hex
        self.job_store.create_job(job_id, config_path.read_text(), str(input_dir), str(output_dir))

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
            "--job-id", job_id,
            *(extra_args or []),
        ]

        container = self.client.containers.run(
            self.image,
            command=command,
            name=CONTAINER_NAME,
            network=self.network,
            environment=credentials.env_vars(),
            volumes=volumes,
            device_requests=[GPU_DEVICE_REQUEST],
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

        self.job_store.ingest_log(job_id, logs)
        self.job_store.finish_job(job_id, exit_code)
        if persist_artifacts:
            self._persist_artifacts(job_id)

        return RunResult(exit_code=exit_code, output_dir=output_dir, logs=logs, job_id=job_id)

    def _persist_artifacts(self, job_id: str) -> None:
        """Copies every dispatch_end event's input_dir/output_dir for this job out of the shared
        workdir volume into self.artifact_store — one shared `alpine` container for the whole
        job (not one per directory: dcr_in/dcr_out alone are per-document subdirectories, so an
        N-document batch would otherwise mean 2N+ separate `docker run`s just for that stage,
        each paying its own container-startup overhead for what's really one bulk copy job).
        Relies entirely on what pipeline.py actually dispatched (via the events those dispatches
        already recorded) rather than a hardcoded list of workdir subdirectory names, so this
        doesn't need updating if pipeline.py's own directory layout changes later.

        shlex.quote() on every path embedded in the shell script is load-bearing, not defensive
        style: `rel` traces back to doc_stem (pipeline.py's per-document crop/dcr/... dirs),
        which comes straight from an uploaded PDF's filename — never sanitized against shell
        metacharacters anywhere upstream. Unquoted string interpolation into `sh -c` here would
        be a real command-injection path from a crafted filename; each argument is quoted as one
        opaque unit regardless of what it contains, closing that off without needing to guess
        which characters are "dangerous"."""
        dispatch_events = self.job_store.get_events(job_id, event="dispatch_end")
        self.artifact_store.root.mkdir(parents=True, exist_ok=True)

        copy_dirs: dict[tuple[str, str], Path] = {}  # (stage, rel) -> dest, dedup across events
        event_dest_paths: dict[int, list[str]] = {}
        for evt in dispatch_events:
            stage = evt.stage or "unknown"
            dest_paths = []
            for container_dir in (evt.input_dir, evt.output_dir):
                rel = relative_to_workdir(container_dir)
                if rel is None:
                    continue
                dest = self.artifact_store.stage_job_dir(stage, job_id) / rel
                dest_paths.append(str(dest))
                copy_dirs[(stage, rel)] = dest
            if dest_paths:
                event_dest_paths[evt.seq] = dest_paths

        if not copy_dirs:
            return

        volumes = {WORKDIR_VOLUME: {"bind": "/w", "mode": "ro"}}
        commands = []
        for i, ((_stage, rel), dest) in enumerate(copy_dirs.items()):
            dest.mkdir(parents=True, exist_ok=True)
            mount_point = f"/out{i}"
            volumes[str(dest)] = {"bind": mount_point, "mode": "rw"}
            commands.append(f"cp -r {shlex.quote(f'/w/{rel}/.')} {shlex.quote(mount_point + '/')}")
        self.client.containers.run(
            "alpine",
            command=["sh", "-c", " && ".join(commands)],
            volumes=volumes,
            remove=True,
        )

        for seq, dest_paths in event_dest_paths.items():
            self.job_store.set_artifact_path(job_id, seq, ";".join(dest_paths))

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
