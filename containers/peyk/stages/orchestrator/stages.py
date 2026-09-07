"""Stage invocation — in-process dispatch via stage_dispatch.call_stage (see that module's own
docstring for the full mechanism). Before docs-personal/new_containerization_strategy.md's step
5, this shelled out to `docker run` against a sibling container over docker-outside-of-docker;
now peyk-orchestrator lives in the same merged image as every stage it dispatches, so a "stage
run" is just a plain in-process function call. stub_fragment() (below) is not config-driven —
there's no `stub:` YAML key (config.py: "there's no 'stub' concept anymore"; StageConfig has no
`stub`/`image` field) — it's a missing-result fallback pipeline.py falls back to per-region when
a dispatch produced no usable output for that region (e.g. a failed OCR/vlm call), so assembly
can still produce a complete document instead of erroring out over one bad region."""
import argparse
import shutil
import sys
import time
from pathlib import Path

import events
import stage_dispatch


class StageDispatchError(RuntimeError):
    pass


def run_docker_stage(
    model: str | None,
    input_dir: Path,
    output_dir: Path,
    extra_args: list[str] | None = None,
    stage_label: str | None = None,
) -> None:
    """`extra_args` must include a leading "--stage <name>" — every caller in pipeline.py/run.py
    already builds this — that's what selects which sibling stage actually runs; everything
    else in extra_args passes straight through to that stage's own argv, unchanged from before.
    No `image` parameter (there used to be one, docs-personal/new_containerization_strategy.md
    step 8): every stage now lives in this same process, and --stage/--role selection was
    already hardcoded per dispatch function in pipeline.py based on `model`/backend, never
    actually derived from an image lookup — the parameter was accepted and silently ignored
    from step 5 onward, kept only so config.py's per-model image maps didn't need to change
    shape. Those maps are gone too now (see config.py's OCR_MODELS/LAYOUT_MODELS/TSR_MODELS).

    stage_label is the logical pipeline stage this dispatch belongs to for traceability purposes
    (e.g. "tsr", "cell_ocr", "table_full", "fullpage") — distinct from --stage/--role, which pick
    the *implementing* backend module (e.g. tsr's `surya` backend still dispatches --stage surya
    --role tsr, but stage_label stays "tsr"). Every pipeline.py/run.py caller passes this
    explicitly; falls back to the raw --stage value only if a caller doesn't (kept optional so
    this isn't a breaking signature change for any other caller)."""
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
    label = stage_label or args.stage

    argv = []
    if model is not None:
        argv += ["--model", model]
    argv += ["--input", str(input_dir), "--output", str(output_dir), *remaining]

    print(f"[peyk-orchestrator] dispatching --stage {args.stage} {' '.join(argv)}", file=sys.stderr)
    events.emit(label, "dispatch_start", model=model, input_dir=str(input_dir), output_dir=str(output_dir))
    t0 = time.perf_counter()
    exit_code = stage_dispatch.call_stage(args.stage, argv)
    duration_s = time.perf_counter() - t0
    print(f"[peyk-orchestrator] --stage {args.stage}: {duration_s:.2f}s", file=sys.stderr)
    events.emit(
        label, "dispatch_end", model=model, duration_s=round(duration_s, 3), exit_code=exit_code,
        input_dir=str(input_dir), output_dir=str(output_dir),
    )
    if exit_code:
        raise StageDispatchError(f"--stage {args.stage} failed (exit {exit_code}); see output above.")


def stub_fragment(
    stage_name: str, label: str, *, doc_stem: str | None = None, region_id: str | None = None, event_stage: str | None = None
) -> str:
    """Emits a traceable "stub" event whenever assembly falls back to a placeholder for a region
    (missing/failed dispatch result) — see pipeline.py's assemble_document call sites, the only
    callers. doc_stem/region_id are optional only for callers that don't have per-region context;
    every real call site does.

    event_stage overrides the event's stage tag when it would otherwise disagree with the
    logical stage_label the actual dispatch used (run_docker_stage's own stage_label parameter):
    stage_name is a display label for the human-readable stub text ("peyk-vlm", "peyk-tsr", ...)
    and doesn't always match 1:1 — e.g. a figure-description stub is stage_name="peyk-vlm" for
    the message but belongs to the "figures" stage for querying, not "vlm"; a full-table stub is
    stage_name="peyk-tsr" but belongs to "table_full" when that's the path that actually ran.
    Defaults to stage_name.removeprefix("peyk-") when the two do agree."""
    events.emit(
        event_stage or stage_name.removeprefix("peyk-"), "stub", label=label, doc_stem=doc_stem, region_id=region_id
    )
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
