"""peyk-surya calling the tsr stage itself — a new capability, not something this container
could do before (see docs-personal/surya/improvement.md's "Before integrating" section,
decision 1). Only used for --role table-full, and only for a crop Q1's sharpness check has
already flagged as needing a split (see run.py) — a sharp crop never triggers this at all.

Dispatched as a plain OS subprocess (`python3 <tsr's own run.py> ...`), not through
stage_dispatch.call_stage's in-process mechanism the way every OTHER nested dispatch in this
image now works (docs-personal/new_containerization_strategy.md step 5) — deliberately, so this
one call can carry a real, enforceable timeout. call_stage mutates GLOBAL interpreter state
(sys.path, sys.modules) under a process-wide lock (see stage_dispatch.py's own docstring); if a
call_stage invocation hangs and we tried to "give up and move on" after a timeout, the abandoned
thread would still be running in the background, still holding that lock and still mutating
shared state — every OTHER stage dispatch afterward, from any thread, would then block forever
waiting for a lock that never releases, turning one stuck table into the entire rest of the
pipeline hanging. Observed live, twice, before this fix: a table dispatched here wedged with no
error and no timeout, and the whole run stalled indefinitely. A plain subprocess has none of
that risk — it can be killed cleanly with no shared state left corrupted, the same safety
property the old docker-in-docker dispatch had (subprocess.run(..., timeout=...) raising
TimeoutExpired), just without the container-spawn overhead: this is a plain `python3` process
inheriting the parent container's own GPU access (GPU visibility in Docker is a container-level
--gpus flag, not per-process), not a new container."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import stage_dispatch

TSR_RUN_PY = stage_dispatch.STAGES_ROOT / "tsr" / "run.py"

# Generous: one crop's TableFormer inference including cold model load, matching the old
# docker-in-docker dispatch's own bound (subprocess.run(..., timeout=600) — see this file's
# git history). Kept at the same value rather than re-derived, since that bound was never what
# failed in practice (nothing has ever taken close to 600s to complete normally) — it only
# exists to catch a genuine wedge and let run.py's own "never fail, fall back to direct
# recognition" fallback take over instead of blocking forever.
TSR_TIMEOUT_SECONDS = 600


def dispatch_tsr(image_path: Path, workdir: Path) -> dict:
    """workdir must be a real subpath of this process's own workdir (i.e. under the same
    directory tree as this container's own --input/--output arguments)."""
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

    # No capture_output: lets the tsr subprocess's own stdout/stderr (model loading, per-crop
    # progress) stream straight through instead of being silently buffered — same reasoning
    # stages/orchestrator/stages.py's own run_docker_stage docstring already documents for why
    # nested dispatch output should be visible live, not swallowed.
    result = subprocess.run(
        [sys.executable, str(TSR_RUN_PY), "--model", "tableformer", "--input", str(in_dir), "--output", str(out_dir)],
        timeout=TSR_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        raise RuntimeError(f"tsr stage dispatch failed (exit {result.returncode})")

    aug_path = out_dir / f"{image_path.stem}_aug.json"
    return json.loads(aug_path.read_text(encoding="utf-8"))
