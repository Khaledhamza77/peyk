#!/usr/bin/env python3
"""Merged worker entrypoint — dispatches to a stage's own run.py `main()` based on --stage, the
same pattern peyk-surya's own run.py already uses for its three roles (see that container).
Each stage's backends/ package and run.py under stages/<stage>/ are copied in unmodified from
their original single-purpose images (containers/peyk-layout, containers/peyk-tsr,
containers/peyk-simple-ocr, containers/peyk-dcr, containers/peyk-vlm) — this file only routes
argv to the right one, no stage behavior changes here. dcr/vlm need no GPU at runtime (pure
pypdfium2 text extraction / remote API calls) but share this image anyway, same as the design
doc's rationale for keeping this one merged worker rather than a GPU/non-GPU split.

--stage can appear anywhere in argv (peyk-orchestrator's stages.py always appends it after
--model/--input/--output as an extra arg, not as argv[1]) — parsed out with parse_known_args
rather than assumed positional, so every other flag (--model, --input, --output, --lang,
--visualize, --watch, --poll-interval, --role, ...) reaches the stage's own unmodified argparse
exactly as before.
"""
import argparse
import runpy
import sys
from pathlib import Path

STAGES = ("layout", "tsr", "ocr", "dcr", "vlm")


def _load_stage_main(stage: str):
    stage_dir = Path(__file__).parent / "stages" / stage
    # Each stage's run.py resolves its sibling `backends` package via sys.path[0] the same way
    # it did as that stage's own single-purpose image's entrypoint (`from backends import
    # BACKENDS`) — inserting the stage's own directory at the front of sys.path here reproduces
    # that exact resolution, instead of merging three incompatible (differently-shaped)
    # backends packages into one shared namespace.
    sys.path.insert(0, str(stage_dir))
    return runpy.run_path(str(stage_dir / "run.py"))["main"]


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage", required=True, choices=STAGES)
    args, remaining = parser.parse_known_args()

    stage_main = _load_stage_main(args.stage)
    sys.argv = [sys.argv[0]] + remaining
    return stage_main()


if __name__ == "__main__":
    raise SystemExit(main())
