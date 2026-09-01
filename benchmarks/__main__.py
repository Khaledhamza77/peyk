"""CLI: run a sweep, or re-render a report from a saved result set.

    python -m benchmarks run --sweep ocr --repeats 3
    python -m benchmarks report --results benchmarks/results/ocr.json --out report.md
    python -m benchmarks cards          # published model-card figures, no Docker needed

`report` and `cards` deliberately require neither Docker nor a GPU, so the accuracy/citation half
of a deck can be regenerated on any machine.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = REPO_ROOT / "benchmarks" / "results"

# A Windows console defaults to cp1252, which cannot encode document names from this project's own
# Arabic sample PDFs -- printing a report that mentions one would die with UnicodeEncodeError
# rather than produce output. Reconfiguring is a no-op wherever stdout is already UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _cmd_run(args: argparse.Namespace) -> int:
    from . import cases
    from .harness import Harness

    try:
        matrix = cases.build(args.sweep)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"[bench] {len(matrix)} case(s) across sweep(s): {', '.join(args.sweep)}")
    print(f"[bench] {args.repeats} measured repeat(s) each, plus one warmup unless --no-warmup")
    harness = Harness(
        input_dir=args.input,
        output_dir=args.output,
        config_dir=args.config_dir,
        image=args.image,
        sample_interval_s=args.sample_interval,
    )
    harness.run_matrix(matrix, repeats=args.repeats, warmup=not args.no_warmup)

    out = Path(args.results or (DEFAULT_RESULTS / f"{'-'.join(args.sweep)}.json"))
    saved = harness.save(out)
    print(f"[bench] wrote {saved}")

    if args.report:
        from . import report

        report_path = report.write(saved, args.report)
        print(f"[bench] wrote {report_path}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from . import report

    if not Path(args.results).exists():
        print(f"error: no such result file: {args.results}", file=sys.stderr)
        return 2
    text = report.render(args.results)
    if args.out:
        out = Path(args.out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}")
    else:
        sys.stdout.write(text)
    return 0


def _cmd_cards(args: argparse.Namespace) -> int:
    from . import report

    sys.stdout.write(report.published_accuracy_table())
    sys.stdout.write("\n")
    sys.stdout.write(report.relevance_caveat_block())
    sys.stdout.write("\n")
    sys.stdout.write(report.observations_block())
    sys.stdout.write("\n")
    sys.stdout.write(report.sources_block())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a sweep and record latency/compute")
    run.add_argument("--sweep", nargs="+", required=True,
                     help="one or more of: layout, ocr, tsr, fullpage, smart-split, born-digital")
    run.add_argument("--input", default=str(REPO_ROOT / "hotstorage" / "input"))
    run.add_argument("--output", default=str(REPO_ROOT / "hotstorage" / "output"))
    run.add_argument("--config-dir", default=str(REPO_ROOT / "hotstorage" / "peyk-config"))
    run.add_argument("--image", default="peyk:dev")
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--no-warmup", action="store_true",
                     help="skip the discarded warmup run (not recommended: the first run after a "
                          "sidecar start pays lazy weight loading and CUDA context setup)")
    run.add_argument("--sample-interval", type=float, default=1.0, help="GPU sampling interval, seconds")
    run.add_argument("--results", default=None, help="where to write the raw result JSON")
    run.add_argument("--report", default=None, help="also render a Markdown report to this path")
    run.set_defaults(func=_cmd_run)

    rep = sub.add_parser("report", help="re-render a report from a saved result set")
    rep.add_argument("--results", required=True)
    rep.add_argument("--out", default=None, help="write here instead of stdout")
    rep.set_defaults(func=_cmd_report)

    cards = sub.add_parser("cards", help="published model-card figures only (no Docker/GPU needed)")
    cards.set_defaults(func=_cmd_cards)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
