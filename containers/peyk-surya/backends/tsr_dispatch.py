"""peyk-surya calling peyk-tsr itself — a new capability, not something this container could
do before (see docs-personal/surya/improvement.md's "Before integrating" section, decision 1).
Only used for --stage table-full, and only for a crop Q1's sharpness check has already flagged
as needing a split (see run.py) — a sharp crop never triggers this at all.

Requires: the host docker socket mounted into this container (see
containers/peyk-orchestrator/pipeline.py's dispatch_table_full_batch and stages.py) and the
docker CLI installed in this image (see Dockerfile). Dispatches peyk-tsr the same way
peyk-orchestrator's own stages.py dispatches every sibling stage container — --volumes-from
ORCHESTRATOR_CONTAINER_NAME (not a host-path bind mount; see that container's own comment for
why: a literal `-v <path>:/data/in` here would resolve against the HOST filesystem through the
shared docker socket, not this container's own filesystem view, silently binding an empty
directory if the paths don't already point at a real host-visible path).
"""
import json
import shutil
import subprocess
from pathlib import Path

# Must match peyk-orchestrator/stages.py's own constants exactly — this dispatch has to join
# the same network and reference the same well-known orchestrator container name that other
# sibling stage containers already do.
PEYK_NETWORK = "peyk-net"
ORCHESTRATOR_CONTAINER_NAME = "peyk-orchestrator-run"
# peyk-tsr:dev was merged into peyk:dev (the merged layout+tsr+ocr worker — see
# docs-personal/new_containerization_strategy.md and containers/peyk/); --stage tsr picks the
# tsr role out of that image the same way peyk-orchestrator's own pipeline.py now does.
TSR_IMAGE = "peyk:dev"


def dispatch_tsr(image_path: Path, workdir: Path) -> dict:
    """workdir must be a real subpath of the shared workdir volume peyk-surya itself was
    launched with --volumes-from ORCHESTRATOR_CONTAINER_NAME for (i.e. under the same
    directory tree as this container's own --input/--output arguments) — not an arbitrary
    tempfile.TemporaryDirectory() path, which would only exist inside peyk-surya's own
    filesystem and be invisible to the peyk-tsr sibling this function launches."""
    # Per-crop unique directory names, not fixed ones: run.py dispatches crops through a
    # ThreadPoolExecutor (DEFAULT_OCR_CONCURRENCY=8) all sharing one workdir, so fixed names
    # would race — mkdir collisions, and one call's rmtree deleting another's in-flight files.
    # image_path.stem is already unique per crop (run.py's scratch_image_path uses it the same
    # way). exist_ok=True as defense in depth.
    in_dir = workdir / f"tsr_dispatch_in_{image_path.stem}"
    out_dir = workdir / f"tsr_dispatch_out_{image_path.stem}"
    for d in (in_dir, out_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)

    shutil.copy(image_path, in_dir / image_path.name)

    # Deterministic per-crop container name + a force-remove sweep before launching, mirroring
    # peyk-orchestrator/stages.py's run_docker_stage: docker-outside-of-docker means killing
    # THIS container does NOT kill the sibling it spawned via the host socket, so an orphan can
    # survive and fight the next run for GPU memory. A fixed name means any such orphan is
    # always found and removed first. Errors ignored — the common case is nothing to remove.
    container_name = f"peyk-tsr-dispatch-{image_path.stem}"
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, text=True)

    subprocess.run(
        [
            "docker", "run", "--rm", "--name", container_name, "--gpus", "all",
            "--network", PEYK_NETWORK,
            "--volumes-from", ORCHESTRATOR_CONTAINER_NAME,
            TSR_IMAGE,
            "--model", "tableformer",
            "--input", str(in_dir),
            "--output", str(out_dir),
            "--stage", "tsr",
        ],
        check=True,
        # Bounded so a hung sibling can't stall the whole batch forever. Generous: one crop's
        # TableFormer inference including cold model load. On expiry the TimeoutExpired
        # propagates and run.py's caller falls back to direct recognition.
        timeout=600,
    )

    aug_path = out_dir / f"{image_path.stem}_aug.json"
    return json.loads(aug_path.read_text(encoding="utf-8"))
