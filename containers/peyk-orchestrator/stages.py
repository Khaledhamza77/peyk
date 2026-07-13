"""Stage invocation. A real stage shells out to `docker run` against a built image; a
stubbed stage (`stub: true` in config, e.g. a stage a given run doesn't need) returns a
placeholder fragment instead, so the dispatch/assembly path can be exercised end to end
without every container necessarily running. Flip `stub: false` and set `image`/`backend`
in the config once a stage's container is ready."""
import shutil
import subprocess
import sys
import time
from pathlib import Path


class DockerStageError(RuntimeError):
    pass

# Named volume for PaddleX's model cache (~/.paddlex/official_models inside the
# container). Both peyk-layout and peyk-simple-ocr's Paddle-based backends download weights
# on first use; without this, every fresh `--rm` container re-downloads them from scratch.
# Docker creates the volume automatically on first use if it doesn't already exist.
#
# peyk-layout's Heron and DocLayout-YOLO backends also fetch weights via huggingface_hub,
# which caches to ~/.cache/huggingface, not ~/.paddlex — but that cache is baked directly
# into the peyk-layout image at build time instead (see that Dockerfile), since neither
# backend has PaddleX's GPU-import constraint blocking a build-time download. Deliberately
# NOT mounting a runtime volume over ~/.cache/huggingface here: doing so would shadow that
# baked-in layer and silently defeat it, re-introducing the every-run re-download this was
# meant to fix.
PADDLEX_CACHE_VOLUME = "peyk-paddlex-cache"

# Shared network so peyk-paddleocr-vl (a per-stage `--rm` container, like every other stage)
# can reach peyk-vllm-paddleocr (a separately-started, persistent sidecar — see
# peyk-vllm-paddleocr/README.md) by container name. Every stage joins this network
# unconditionally rather than only the ocr stage, since it's harmless for stages that don't
# need it and keeps this function's branching simple.
PEYK_NETWORK = "peyk-net"

# Must match the --name run_local.sh gives the orchestrator container itself. Every sibling
# stage container inherits its mounts via `--volumes-from` (below) instead of bind-mounting
# input_dir/output_dir by host path — see run_docker_stage's docstring for why a host-path
# bind mount doesn't work here.
ORCHESTRATOR_CONTAINER_NAME = "peyk-orchestrator-run"


def _stage_container_name(image: str) -> str:
    """A stable, predictable name per stage type (e.g. "peyk-stage-ocr"), not per invocation.
    This is what makes cleanup below possible: docker-outside-of-docker means killing the
    orchestrator's own container does NOT kill a sibling stage container it launched via the
    host socket (it's a separate container on the same host daemon, not a child process) — a
    real zombie-container/GPU-memory leak we hit in practice. A fixed name means any orphan
    left behind by a previous killed run is always found and removed before the next one
    starts, instead of silently piling up and fighting the new run for GPU memory."""
    return f"peyk-stage-{image.split(':')[0].removeprefix('peyk-')}"


def run_docker_stage(
    image: str,
    model: str | None,
    input_dir: Path,
    output_dir: Path,
    extra_args: list[str] | None = None,
    extra_docker_args: list[str] | None = None,
    gpu: bool = True,
) -> None:
    # Cleared rather than just mkdir(exist_ok=True): callers match results back by reading
    # every *.json this stage writes, so a stale file left over from an earlier run at the
    # same path (e.g. a region index no longer dispatched to this stage) would silently be
    # picked up as if it belonged to the current run.
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    container_name = _stage_container_name(image)
    t_cleanup = time.perf_counter()
    # Force-remove any container left over from a previous run under this same stage name
    # (e.g. orphaned by `docker kill` on the orchestrator without also killing this sibling —
    # see _stage_container_name docstring). Errors ignored: the common case is there's
    # nothing to remove.
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, text=True)
    # Idempotent: errors (already exists) ignored. Must exist before peyk-vllm-paddleocr is
    # started too (see that container's start.sh) — whichever comes up first creates it.
    subprocess.run(["docker", "network", "create", PEYK_NETWORK], capture_output=True, text=True)
    cleanup_s = time.perf_counter() - t_cleanup

    cmd = ["docker", "run", "--rm", "--name", container_name, "--network", PEYK_NETWORK]
    if gpu:
        cmd += ["--gpus", "all"]
    cmd += [
        # Inherits peyk-orchestrator's own mounts (its /hotstorage bind mount, specifically)
        # instead of bind-mounting input_dir/output_dir by the path this orchestrator
        # *container* sees them at. This stage is dispatched via docker-outside-of-docker —
        # the docker CLI call below reaches the HOST's docker daemon through the shared
        # socket, which resolves any bind-mount source against the host filesystem, not the
        # orchestrator container's own filesystem. A literal `-v {input_dir}:/input` (where
        # input_dir is e.g. "/hotstorage/layout_out" as seen inside the orchestrator) would
        # make the host daemon look for that exact path on the host/VM root and silently bind
        # an empty or nonexistent directory instead — hit in practice: peyk-layout reported
        # "no PDF/image files found in /input" because its /input mount resolved to nothing.
        # --volumes-from sidesteps host-path translation entirely: Docker already knows the
        # real mount behind ORCHESTRATOR_CONTAINER_NAME's /hotstorage, so as long as
        # input_dir/output_dir are always subpaths of it (true here — see pipeline.py/run.py's
        # DEFAULT_WORKDIR), both containers see identical files at identical paths regardless
        # of host OS or Docker Desktop's own path-mapping quirks.
        "--volumes-from", ORCHESTRATOR_CONTAINER_NAME,
        "-v", f"{PADDLEX_CACHE_VOLUME}:/root/.paddlex",
    ]
    if extra_docker_args:
        # peyk-vlm's cloud credential files (containers/peyk-vlm/.env, gcp-key.json) are fixed
        # repo-local paths, not workdir paths inside the orchestrator's own mounts — passed as
        # real host-absolute paths here rather than routed through --volumes-from (which only
        # re-derives paths the orchestrator container itself already has). See pipeline.py's
        # _vlm_credential_docker_args and run_local.sh's PEYK_VLM_ENV_FILE/PEYK_VLM_GCP_KEY_FILE.
        cmd += extra_docker_args
    cmd += [image]
    if model is not None:
        cmd += ["--model", model]
    cmd += [
        "--input", str(input_dir),
        "--output", str(output_dir),
    ]
    if extra_args:
        cmd += extra_args
    print(f"[peyk-orchestrator] {' '.join(cmd)}", file=sys.stderr)
    # No capture_output: that would swallow the child container's own stdout/stderr into
    # Python variables instead of letting it stream to the terminal live (model loading,
    # per-crop progress, etc. would only ever surface after the fact, inside the exception
    # message below, and only on failure — silently discarded on success). Inheriting the
    # parent's stdout/stderr instead means it's visible in real time, same as running that
    # `docker run` directly, at the cost of not being able to embed it in the error message.
    t_run = time.perf_counter()
    result = subprocess.run(cmd)
    run_s = time.perf_counter() - t_run
    print(
        f"[peyk-orchestrator] {image} stage: cleanup (docker rm/network create) {cleanup_s:.2f}s, "
        f"docker run {run_s:.2f}s",
        file=sys.stderr,
    )
    if result.returncode != 0:
        raise DockerStageError(f"{image} failed (exit {result.returncode}); see output above.")


def stub_fragment(stage_name: str, label: str) -> str:
    return f"*[stub: `{stage_name}` not yet built — {label} region skipped]*"


def list_vlm_models(image: str) -> dict[str, str]:
    """Queries peyk-vlm's own MODEL_REGISTRY directly (`--list-models`, "<key>\\t<provider>" per
    line) rather than config.py guessing which models peyk-vlm supports — and which cloud each
    one's credentials need — from a naming convention (e.g. assuming every key starts with
    "bedrock-"/"vertex-"). Real ground truth, not an assumption that could silently drift out of
    sync with the registry. No --volumes-from/--network needed: this doesn't touch any mounted
    dir or need to reach a sidecar, just prints a static list and exits. Returns
    {model_key: provider}."""
    result = subprocess.run(["docker", "run", "--rm", image, "--list-models"], capture_output=True, text=True)
    if result.returncode != 0:
        raise DockerStageError(
            f"{image} --list-models failed (exit {result.returncode}): {result.stderr.strip()} — "
            "is the image built? (docker build -t peyk-vlm:dev containers/peyk-vlm)"
        )
    models = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        key, provider = line.split("\t")
        models[key] = provider
    return models
