"""Stage invocation — in-process dispatch via stage_dispatch.call_stage (see that module's own
docstring for the full mechanism). Before docs-personal/new_containerization_strategy.md's step
5, this shelled out to `docker run` against a sibling container over docker-outside-of-docker;
now peyk-orchestrator lives in the same merged image as every stage it dispatches, so a "stage
run" is just a plain in-process function call. A stubbed stage (`stub: true` in config, e.g. a
stage a given run doesn't need) returns a placeholder fragment instead, so the dispatch/assembly
path can be exercised end to end without every stage necessarily running. Flip `stub: false` and
set `image`/`backend` in the config once a stage is ready."""
import argparse
import shutil
import sys
import time
from pathlib import Path

import stage_dispatch


class StageDispatchError(RuntimeError):
    pass


def run_docker_stage(
    model: str | None,
    input_dir: Path,
    output_dir: Path,
    extra_args: list[str] | None = None,
) -> None:
    """`extra_args` must include a leading "--stage <name>" — every caller in pipeline.py/run.py
    already builds this — that's what selects which sibling stage actually runs; everything
    else in extra_args passes straight through to that stage's own argv, unchanged from before.
    No `image` parameter (there used to be one, docs-personal/new_containerization_strategy.md
    step 8): every stage now lives in this same process, and --stage/--role selection was
    already hardcoded per dispatch function in pipeline.py based on `model`/backend, never
    actually derived from an image lookup — the parameter was accepted and silently ignored
    from step 5 onward, kept only so config.py's per-model image maps didn't need to change
    shape. Those maps are gone too now (see config.py's OCR_MODELS/LAYOUT_MODELS/TSR_MODELS)."""
    # Cleared rather than just mkdir(exist_ok=True): callers match results back by reading
    # every *.json this stage writes, so a stale file left over from an earlier run at the
    # same path (e.g. a region index no longer dispatched to this stage) would silently be
    # picked up as if it belonged to the current run.
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage", required=True)
    args, remaining = parser.parse_known_args(extra_args or [])

    argv = []
    if model is not None:
        argv += ["--model", model]
    argv += ["--input", str(input_dir), "--output", str(output_dir), *remaining]

    print(f"[peyk-orchestrator] dispatching --stage {args.stage} {' '.join(argv)}", file=sys.stderr)
    t0 = time.perf_counter()
    exit_code = stage_dispatch.call_stage(args.stage, argv)
    print(f"[peyk-orchestrator] --stage {args.stage}: {time.perf_counter() - t0:.2f}s", file=sys.stderr)
    if exit_code:
        raise StageDispatchError(f"--stage {args.stage} failed (exit {exit_code}); see output above.")


def stub_fragment(stage_name: str, label: str) -> str:
    return f"*[stub: `{stage_name}` not yet built — {label} region skipped]*"


def list_vlm_models() -> dict[str, str]:
    """Queries the vlm stage's own MODEL_REGISTRY directly (`--list-models`, "<key>\\t<provider>"
    per line) rather than config.py guessing which models it supports — and which cloud each
    one's credentials need — from a naming convention (e.g. assuming every key starts with
    "bedrock-"/"vertex-"). Real ground truth, not an assumption that could silently drift out of
    sync with the registry. In-process now: captures the vlm stage's own stdout instead of a
    subprocess's, same parsing either way."""
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exit_code = stage_dispatch.call_stage("vlm", ["--list-models"])
    if exit_code:
        raise StageDispatchError(f"vlm --list-models failed (exit {exit_code}); see output above.")
    models = {}
    for line in buf.getvalue().splitlines():
        if not line.strip():
            continue
        key, provider = line.split("\t")
        models[key] = provider
    return models
