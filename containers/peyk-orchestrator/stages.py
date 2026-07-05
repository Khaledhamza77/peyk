"""Stage invocation. A real stage shells out to `docker run` against a built image; a
stubbed stage (no image built yet — peyk-dcr/peyk-tsr/peyk-vlm as of this writing) returns
a placeholder fragment instead, so the dispatch/assembly path can be exercised end to end
before every container exists. Flip `stub: false` and set `image`/`backend` in the config
once a stage's container is ready."""
import shutil
import subprocess
import sys
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
    model: str,
    input_dir: Path,
    output_dir: Path,
    extra_args: list[str] | None = None,
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
    # Force-remove any container left over from a previous run under this same stage name
    # (e.g. orphaned by `docker kill` on the orchestrator without also killing this sibling —
    # see _stage_container_name docstring). Errors ignored: the common case is there's
    # nothing to remove.
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, text=True)
    # Idempotent: errors (already exists) ignored. Must exist before peyk-vllm-paddleocr is
    # started too (see that container's start.sh) — whichever comes up first creates it.
    subprocess.run(["docker", "network", "create", PEYK_NETWORK], capture_output=True, text=True)

    cmd = ["docker", "run", "--rm", "--name", container_name, "--network", PEYK_NETWORK]
    if gpu:
        cmd += ["--gpus", "all"]
    cmd += [
        "-v", f"{input_dir.resolve()}:/input:ro",
        "-v", f"{output_dir.resolve()}:/output",
        "-v", f"{PADDLEX_CACHE_VOLUME}:/root/.paddlex",
        image,
        "--model", model,
        "--input", "/input",
        "--output", "/output",
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
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise DockerStageError(f"{image} failed (exit {result.returncode}); see output above.")


def stub_fragment(stage_name: str, label: str) -> str:
    return f"*[stub: `{stage_name}` not yet built — {label} region skipped]*"
