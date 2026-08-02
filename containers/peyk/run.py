#!/usr/bin/env python3
"""Merged worker entrypoint — dispatches to a stage's own run.py `main()` based on --stage, via
stage_dispatch.call_stage (see that module's own docstring for the in-process dispatch
mechanism). Default stage is "orchestrator" (docs-personal/new_containerization_strategy.md
step 5): a plain `docker run peyk:dev --config ... --input ... --output ...` now runs the whole
pipeline, the same way `peyk-orchestrator`'s own CLI used to before its code moved to
stages/orchestrator/ and its dispatch logic became in-process calls instead of nested
`docker run`s. --stage <name> is still available to invoke any single stage directly (debugging,
the isolated-stage tests documented in the consolidation plan).

Each stage's backends/ package and run.py under stages/<stage>/ are copied in unmodified from
their original single-purpose images (containers/peyk-layout, containers/peyk-tsr,
containers/peyk-simple-ocr, containers/peyk-dcr, containers/peyk-vlm, containers/peyk-surya,
containers/peyk-paddleocr-vl, containers/peyk-orchestrator) — this file only routes argv to the
right one, no stage behavior changes here.

--stage can appear anywhere in argv (peyk-orchestrator's stages.py always appends it after
--model/--input/--output as an extra arg, not as argv[1]) — parsed out with parse_known_args
rather than assumed positional, so every other flag (--model, --input, --output, --lang,
--visualize, --watch, --poll-interval, --role, --config, ...) reaches the stage's own unmodified
argparse exactly as before. --stage only ever picks the module; a module with more than one
internal mode of its own (vlm, surya) uses its own --role flag one level down to pick that —
never --stage again, to avoid exactly the collision surya's own run.py used to have before its
CLI's former --stage argument was renamed to --role for this merge (see that file's own
comment).
"""
import argparse

import stage_dispatch

STAGES = ("layout", "tsr", "ocr", "dcr", "vlm", "surya", "paddleocr-vl", "orchestrator")


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage", default="orchestrator", choices=STAGES)
    args, remaining = parser.parse_known_args()
    return stage_dispatch.call_stage(args.stage, remaining)


if __name__ == "__main__":
    raise SystemExit(main())
