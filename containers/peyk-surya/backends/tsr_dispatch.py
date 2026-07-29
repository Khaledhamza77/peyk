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
TSR_IMAGE = "peyk-tsr:dev"


def dispatch_tsr(image_path: Path, workdir: Path) -> dict:
    """workdir must be a real subpath of the shared workdir volume peyk-surya itself was
    launched with --volumes-from ORCHESTRATOR_CONTAINER_NAME for (i.e. under the same
    directory tree as this container's own --input/--output arguments) — not an arbitrary
    tempfile.TemporaryDirectory() path, which would only exist inside peyk-surya's own
    filesystem and be invisible to the peyk-tsr sibling this function launches."""
    in_dir = workdir / "tsr_dispatch_in"
    out_dir = workdir / "tsr_dispatch_out"
    for d in (in_dir, out_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)

    shutil.copy(image_path, in_dir / image_path.name)

    subprocess.run(
        [
            "docker", "run", "--rm", "--gpus", "all",
            "--network", PEYK_NETWORK,
            "--volumes-from", ORCHESTRATOR_CONTAINER_NAME,
            TSR_IMAGE,
            "--model", "tableformer",
            "--input", str(in_dir),
            "--output", str(out_dir),
        ],
        check=True,
    )

    aug_path = out_dir / f"{image_path.stem}_aug.json"
    return json.loads(aug_path.read_text(encoding="utf-8"))
