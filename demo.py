#!/usr/bin/env python3
"""End-to-end demo of the peyk SDK (sdk/) — the Python equivalent of running
containers/peyk-vllm-surya/start.sh + containers/peyk-vllm-paddleocr/start.sh +
containers/peyk/run_local.sh by hand. Run `uv sync` from the repo root once first to create the
repo-level .venv this script expects to run under.

Requires Docker with GPU support, and the peyk:dev image already built (or pass --build-image to
build it from containers/peyk here). Defaults mirror
containers/peyk/stages/orchestrator/config/example.yaml's own model choices.

Usage:
    python demo.py                       # uses hotstorage/input -> hotstorage/output
    python demo.py --build-image         # docker build peyk:dev from containers/peyk first
    python demo.py --input data --output hotstorage/output --skip-sidecars
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.resolve()

sys.path.insert(0, str(REPO_ROOT / "sdk" / "src"))
from peyk import Peyk, PipelineConfig, StageConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=REPO_ROOT / "hotstorage" / "input", type=Path)
    parser.add_argument("--output", default=REPO_ROOT / "hotstorage" / "output", type=Path)
    parser.add_argument("--config-dir", default=REPO_ROOT / "hotstorage" / "peyk-config", type=Path)
    parser.add_argument("--image", default="peyk:dev")
    parser.add_argument("--build-image", action="store_true", help="docker build peyk:dev from containers/peyk first")
    parser.add_argument("--skip-sidecars", action="store_true", help="assume the vLLM sidecars are already running")
    parser.add_argument("--bedrock-token", default=None, help="overrides containers/peyk/.env's AWS_BEARER_TOKEN_BEDROCK")
    parser.add_argument("--gcp-key", default=None, type=Path, help="overrides containers/peyk/gcp-key.json")
    parser.add_argument(
        "--persist-artifacts", action="store_true",
        help="also copy this run's crops/per-region model output into peyk.artifacts (see the job history section below)",
    )
    return parser.parse_args()


def _default_credential(explicit, fallback_path: Path):
    if explicit is not None:
        return explicit
    return fallback_path if fallback_path.exists() else None


def main() -> int:
    args = parse_args()

    peyk = Peyk(image=args.image)

    if args.build_image:
        print(f"Building {args.image} from containers/peyk ...")
        peyk.build_image(REPO_ROOT / "containers" / "peyk", tag=args.image)

    peyk.configure(
        PipelineConfig(
            layout=StageConfig(model="heron"),
            tsr=StageConfig(model="surya"),
            ocr=StageConfig(model="paddleocr-vl", lang="arabic"),
            cell_ocr=StageConfig(model="surya"),
            figures=StageConfig(model="gemini-3-1-flash-lite"),
        ),
        config_dir=args.config_dir,
    )

    bedrock_env = REPO_ROOT / "containers" / "peyk" / ".env"
    bedrock_token = args.bedrock_token
    if bedrock_token is None and bedrock_env.exists():
        for line in bedrock_env.read_text().splitlines():
            if line.startswith("AWS_BEARER_TOKEN_BEDROCK="):
                bedrock_token = line.split("=", 1)[1].strip()
    gcp_key = _default_credential(args.gcp_key, REPO_ROOT / "containers" / "peyk" / "gcp-key.json")
    peyk.set_credentials(bedrock_bearer_token=bedrock_token, gcp_key_path=gcp_key)

    if args.skip_sidecars:
        print("Skipping sidecar startup (--skip-sidecars) — assuming they're already running.")
    else:
        needed = peyk.ensure_sidecars(wait=True)
        print(f"Sidecars ready: {needed or '(none needed for this config)'}")

    print(f"Running peyk:dev over {args.input} -> {args.output} ...")
    result = peyk.run(input_dir=args.input, output_dir=args.output, persist_artifacts=args.persist_artifacts)

    print(result.logs)
    print(f"Exit code: {result.exit_code}")
    print(f"Output written to: {result.output_dir}")

    # Job history — recorded automatically by every run() call, regardless of --persist-artifacts.
    # See docs-personal/central_logging_system.md and sdk/README.md's "Job history & artifacts".
    print(f"\nJob {result.job_id} ({peyk.jobs.get_job(result.job_id).status}) — stage timeline:")
    for event in peyk.jobs.get_events(result.job_id):
        detail = f"{event.duration_s:.2f}s" if event.duration_s is not None else (event.message or "")
        print(f"  [{event.stage or '?':>10}] {event.event or '?':<14} {detail}")
        if event.artifact_path:
            print(f"  {'':>10}   artifacts: {event.artifact_path}")

    if args.persist_artifacts:
        print(f"\nArtifacts persisted under {peyk.artifacts.root}/<stage>/{result.job_id}/ for each stage above.")
        print(f"Clean them up later with e.g. peyk.artifacts.cleanup(job_id={result.job_id!r})")

    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
