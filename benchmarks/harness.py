"""Drives a matrix of pipeline configurations and records what each one cost to run.

Scope, deliberately: this measures **latency and computational cost only** -- wall time, per-stage
time, GPU memory/utilisation/power, and the work-unit counts needed to normalise them. It does not
score output quality, and it must not be presented as if it did. This project has no ground-truth
corpus and no annotated test set, so any "accuracy" number produced here would be invented.
Recognition quality is evidenced two other ways: each model's own published benchmark figures
(cited to the model card -- see model_cards.py) and the demo's own side-by-side output examples.

What is measured here needs no ground truth at all, which is exactly why it is the part worth
measuring ourselves.

Built on the traceability events the orchestrator already emits
(containers/peyk/stages/orchestrator/events.py) and the SQLite JobStore the SDK already ingests
them into (sdk/src/peyk/history.py) -- per-stage `duration_s` is read back by job_id rather than
re-derived by scraping stderr, so this harness does not carry its own copy of the pipeline's
timing logic.
"""
from __future__ import annotations

import json
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import docker
from peyk import JobStore, Peyk, PipelineConfig

from . import gpu, workunits

# How far a single run's GPU baseline may sit from its case's median before that run is excluded
# from the VRAM aggregate. Sized to admit ordinary sampling jitter (tens of MiB) while rejecting
# the multi-GB baseline shifts seen when an unrelated process releases or claims the card.
BASELINE_TOLERANCE_MIB = 250.0


@dataclass
class BenchmarkCase:
    """One point in the sweep. `name` is the row label that ends up on the slide, so make it read
    as the thing being compared ("ocr=tesseract", "fullpage=gemini-3-1-flash-lite"), not as a
    filename."""

    name: str
    config: PipelineConfig
    # Free-text, carried through to the report -- e.g. "CPU-only backend" or "managed API, no
    # local GPU cost". Worth filling in: it is what stops a reader misreading a 0 MiB VRAM row as
    # a measurement failure.
    note: str = ""


@dataclass
class StageTiming:
    stage: str
    model: str | None
    duration_s: float
    started_at: float
    ended_at: float
    exit_code: int | None
    # Whole-board GPU stats for this stage's own time window, sliced out of the run-wide trace.
    gpu: dict = field(default_factory=dict)


@dataclass
class RunRecord:
    case: str
    repeat: int
    job_id: str
    exit_code: int
    wall_s: float
    warmup: bool
    stages: list[StageTiming] = field(default_factory=list)
    gpu: dict = field(default_factory=dict)
    units: dict = field(default_factory=dict)
    note: str = ""
    error: str | None = None

    def stage_total_s(self) -> float:
        return sum(s.duration_s for s in self.stages)

    def to_dict(self) -> dict:
        out = asdict(self)
        return out


def environment_fingerprint(client: "docker.DockerClient | None" = None, image: str = "peyk:dev") -> dict:
    """Recorded once per report. A latency number is only comparable against another taken on the
    same hardware, the same image, and the same commit -- so all three are captured rather than
    left to be remembered later."""
    fingerprint = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": platform.platform(),
        "python": platform.python_version(),
        "gpu": gpu.gpu_name(),
        "gpu_driver": gpu.driver_version(),
    }
    try:
        fingerprint["git_sha"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout.strip() or None
        fingerprint["git_dirty"] = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=10, check=False,
            ).stdout.strip()
        )
    except (subprocess.TimeoutExpired, OSError):
        pass
    if client is not None:
        try:
            # The image actually under test, not a hardcoded tag -- a sweep run with --image
            # against a different build must not record the default one's digest.
            found = client.images.get(image)
            fingerprint["image_id"] = found.id
            fingerprint["image_tags"] = found.tags
        except docker.errors.DockerException:
            pass
    return fingerprint


class Harness:
    def __init__(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        config_dir: str | Path,
        image: str = "peyk:dev",
        sample_interval_s: float = 1.0,
        db_path: str | Path | None = None,
    ):
        self.input_dir = Path(input_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.config_dir = Path(config_dir).resolve()
        self.image = image
        self.sample_interval_s = sample_interval_s
        # db_path must be handed to Peyk, not used to build a second JobStore here: PeykRunner
        # writes events through the store Peyk constructed, so an independent store on a
        # different path would read back empty and every run would silently report zero stages.
        self.peyk = Peyk(image=image, db_path=db_path) if db_path is not None else Peyk(image=image)
        self.client = self.peyk.client
        self.jobs: JobStore = self.peyk.jobs
        self.records: list[RunRecord] = []
        # Which sidecars are currently up, so run_case does not restart them between repeats --
        # see _ensure_sidecars().
        self._active_sidecars: set[str] = set()
        self._resolve_credentials()

    def _resolve_credentials(self) -> None:
        """Picks up Bedrock/GCP credentials the same way demo.py and run_local.sh do.

        Without this, every managed-VLM case fails at dispatch -- including cases that look purely
        self-hosted, because the reference config still routes `figures` to a VLM (there is no
        self-hosted figures backend). The token is read straight from disk into the Credentials
        object and never logged or written into any result file.
        """
        repo_root = Path(__file__).resolve().parent.parent
        bedrock_token = None
        env_file = repo_root / "containers" / "peyk" / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("AWS_BEARER_TOKEN_BEDROCK="):
                    bedrock_token = line.split("=", 1)[1].strip()
                    break
        gcp_key = repo_root / "containers" / "peyk" / "gcp-key.json"
        self.peyk.set_credentials(
            bedrock_bearer_token=bedrock_token,
            gcp_key_path=gcp_key if gcp_key.exists() else None,
        )
        found = [n for n, ok in (("bedrock", bedrock_token), ("gcp", gcp_key.exists())) if ok]
        print(f"[bench] credentials resolved: {', '.join(found) or 'none found'}", flush=True)

    # ------------------------------------------------------------------ internals

    def _stage_timings(self, job_id: str, trace: gpu.GpuTrace) -> list[StageTiming]:
        """Pairs each dispatch_start with its matching dispatch_end and slices the GPU trace to
        that window.

        Pairing is per-stage FIFO rather than by any shared id, because events.emit() does not
        carry one -- within a single stage, dispatches are strictly sequential (pipeline.py
        dispatches one batch per stage, in order), so the Nth start belongs to the Nth end.

        Event timestamps are container-side time.time(); a Linux container shares the host clock,
        so they are directly comparable to the host-side GPU sample timestamps. That is what makes
        per-stage GPU attribution possible at all without instrumenting inside each backend.
        """
        events = self.jobs.get_events(job_id)
        pending: dict[str, list] = {}
        timings: list[StageTiming] = []
        for evt in events:
            if evt.event == "dispatch_start":
                pending.setdefault(evt.stage or "unknown", []).append(evt)
            elif evt.event == "dispatch_end":
                queue = pending.get(evt.stage or "unknown", [])
                start_evt = queue.pop(0) if queue else None
                duration = evt.duration_s if evt.duration_s is not None else 0.0
                # Prefer the start event's own timestamp; fall back to back-dating the end
                # timestamp by the reported duration when a start went missing (truncated log).
                started = start_evt.ts if start_evt is not None and start_evt.ts else (evt.ts or 0.0) - duration
                ended = evt.ts or (started + duration)
                timings.append(
                    StageTiming(
                        stage=evt.stage or "unknown",
                        model=evt.model,
                        duration_s=duration,
                        started_at=started,
                        ended_at=ended,
                        exit_code=evt.exit_code,
                        gpu=trace.window(started, ended).summary(),
                    )
                )
        return timings

    def _ensure_sidecars(self) -> None:
        """Start only the sidecars this config needs that are not already up.

        SidecarManager.start() force-removes and recreates the container it is given, so calling
        peyk.ensure_sidecars() once per run would re-pay the full cold start every single repeat
        -- roughly 14 minutes for Surya on this project's own dev card. That would make the warmup
        run pointless (nothing is ever warm) and every measured repeat a cold-start measurement.

        Sidecars are therefore started once and left up across repeats, and across consecutive
        cases that need the same set. They are also deliberately started *outside* the sampling
        window: their load cost is a one-time deployment cost, not part of per-document latency.
        Their VRAM reservation is still captured -- as the sampler's baseline, since it is already
        resident by the time sampling begins.

        `_active_sidecars` alone is not enough to decide this, though -- it is in-process state,
        empty on every fresh Harness/CLI invocation regardless of what is actually running in
        Docker. Without checking real state, a brand-new process whose first case needs Surya
        would call sidecars.start('surya') unconditionally on a sidecar a previous run (or a
        person, by hand) already brought up and warmed -- discarding it and repaying the full
        cold start for nothing.

        Checked via the container's own running status, not sidecars.is_ready() (an HTTP GET
        against localhost:<port>). Confirmed in practice: a paddleocr-vl sidecar started by hand
        via containers/peyk-vllm-paddleocr/start.sh (rather than through this SDK) publishes no
        host port at all -- SidecarManager's own docstring notes this is deliberate upstream
        behavior it deviates from, precisely so its *own*-started sidecars can be host-polled.
        is_ready() correctly reports False for a start.sh-launched container, since it genuinely
        is not reachable from the host that way -- but it does not need to be: the pipeline
        reaches it container-to-container over peyk-net by container name, never via the host
        port. Trusting is_ready() here would have force-removed (SidecarManager.start()'s first
        step) and replaced a perfectly good, already-warm sidecar the user had started themselves,
        for a check that was never actually testing the thing that matters."""
        needed = self.peyk._config.sidecar_requirements() if self.peyk._config else set()
        for name in needed - self._active_sidecars:
            try:
                container = self.peyk.client.containers.get(self.peyk.sidecars._spec(name).container_name)
                already_running = container.status == "running"
            except Exception:
                already_running = False
            if already_running:
                print(f"[bench]   sidecar {name} already up, reusing", flush=True)
                self._active_sidecars.add(name)

        missing = needed - self._active_sidecars
        for name in missing:
            print(f"[bench]   starting sidecar {name} (cold start, not measured)", flush=True)
            self.peyk.sidecars.start(name)
        if missing:
            for name in missing:
                self.peyk.sidecars.wait_ready(name)
        self._active_sidecars |= needed

    def _run_once(self, case: BenchmarkCase, repeat: int, warmup: bool) -> RunRecord:
        error: str | None = None
        exit_code, job_id = 1, ""
        wall_s = 0.0
        trace = gpu.GpuTrace()

        try:
            # Inside the try: a config that fails validation, or a sidecar that will not come
            # ready, must fail this one case and let run_matrix continue to the next -- not abort
            # the whole sweep after it has already spent GPU time on earlier cases.
            self.peyk.configure(case.config, config_dir=self.config_dir)
            self._ensure_sidecars()

            with gpu.GpuSampler(interval_s=self.sample_interval_s) as sampler:
                t0 = time.perf_counter()
                try:
                    result = self.peyk.run(input_dir=self.input_dir, output_dir=self.output_dir)
                    exit_code, job_id = result.exit_code, result.job_id
                    if exit_code != 0:
                        # docker-py does not raise on a non-zero exit -- the container ran fine
                        # as far as Python is concerned, its own process just returned failure.
                        # That is the common case for a real pipeline bug (an unhandled exception
                        # inside run.py), and without this the failure would surface here only as
                        # a bare exit code, with the actual traceback sitting unread in
                        # result.logs. Tail rather than the whole log: a multi-document batch run
                        # can produce megabytes of per-crop progress lines before the traceback.
                        tail = "\n".join(result.logs.splitlines()[-40:])
                        error = f"container exited {exit_code}\n{tail}"
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                # Measured before the `with` closes: GpuSampler.__exit__ joins the sampling
                # thread, which can block for up to 3x the sampling interval. Timing outside the
                # block would add that to every single measurement as a constant.
                wall_s = time.perf_counter() - t0
            trace = sampler.trace
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        return RunRecord(
            case=case.name,
            repeat=repeat,
            job_id=job_id,
            exit_code=exit_code,
            wall_s=round(wall_s, 3),
            warmup=warmup,
            stages=self._stage_timings(job_id, trace) if job_id else [],
            gpu=trace.summary(),
            units=self._work_units(job_id) if job_id else {},
            note=case.note,
            error=error,
        )

    def _work_units(self, job_id: str) -> dict:
        """Counts only the workdir directories THIS job actually dispatched against.

        The workdir volume is deliberately persistent (it is what makes intermediate crops
        inspectable between runs), so it accumulates directories from earlier runs under
        different configs. Counting the whole volume would fold those into this run's totals --
        confirmed in practice: a volume left over from earlier work carried a populated
        `fullpage_in` from a run that used fullpage mode, which would be silently attributed to a
        later per-region run that never touched it.

        The job's own dispatch events name every directory it used, so they are the correct
        filter."""
        touched: set[str] = set()
        for evt in self.jobs.get_events(job_id, event="dispatch_end"):
            for container_dir in (evt.input_dir, evt.output_dir):
                if container_dir:
                    touched.add(container_dir.replace("\\", "/").rstrip("/").split("/")[-1])
        return workunits.count(self.client, only_dirs=touched or None).to_dict()

    # ------------------------------------------------------------------ public API

    def run_case(self, case: BenchmarkCase, repeats: int = 3, warmup: bool = True) -> list[RunRecord]:
        """One warmup run (recorded but flagged, never aggregated) followed by `repeats` measured
        runs.

        The warmup is not optional politeness -- the first run after a sidecar starts pays lazy
        weight loading, PaddleX sub-model downloads into the cache volume, and CUDA context setup.
        Including it would inflate that config's median by an amount that has nothing to do with
        steady-state cost. It is kept in the output rather than discarded so the cold/warm gap is
        itself reportable, which is a genuinely interesting number for a deployment discussion.
        """
        produced: list[RunRecord] = []
        if warmup:
            produced.append(self._run_once(case, repeat=-1, warmup=True))
        for i in range(repeats):
            produced.append(self._run_once(case, repeat=i, warmup=False))
        self.records.extend(produced)
        return produced

    def run_matrix(self, cases: list[BenchmarkCase], repeats: int = 3, warmup: bool = True) -> list[RunRecord]:
        for case in cases:
            print(f"[bench] case {case.name!r}: warmup={warmup} repeats={repeats}", flush=True)
            for record in self.run_case(case, repeats=repeats, warmup=warmup):
                tag = "warmup" if record.warmup else f"run {record.repeat}"
                status = "ok" if record.exit_code == 0 else f"FAILED ({record.error or record.exit_code})"
                print(f"[bench]   {tag}: {record.wall_s:.1f}s {status}", flush=True)
        return self.records

    def save(self, path: str | Path) -> Path:
        """Writes every raw record plus the environment fingerprint. Scoring/aggregation reads
        this file rather than re-running anything -- report definitions change far more often than
        it is worth re-paying GPU time for."""
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "environment": environment_fingerprint(self.client, image=self.image),
            "records": [r.to_dict() for r in self.records],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path


def failures(records: list[RunRecord]) -> dict[str, dict]:
    """Cases with zero successful measured runs -- the complement of aggregate() below.

    A case whose every measured run fails identically (the same config, the same corpus, a
    deterministic bug) never gets a key in aggregate()'s summary, since that function only ever
    looks at successful runs. Left there, such a case simply vanishes from every report table --
    indistinguishable from a case nobody ran. Found in practice: layout=doclayout-yolo paired with
    tsr=tableformer failed its warmup and first measured run identically
    ("x1 must be greater than or equal to x0", a real geometry bug in
    stages/tsr/backends/base.py's col_boxes(), unrelated to this harness) and would otherwise have
    disappeared from the latency table with no indication it was ever attempted.

    Returns {case: {"attempts": n, "errors": [distinct error strings]}} for every case where every
    non-warmup run failed. A case with at least one success is not "failed" even if some repeats
    errored -- that partial-failure signal belongs in the successful case's own row (a future
    improvement), not here.
    """
    by_case: dict[str, list[RunRecord]] = {}
    for record in records:
        if record.warmup:
            continue
        by_case.setdefault(record.case, []).append(record)

    out: dict[str, dict] = {}
    for case_name, runs in by_case.items():
        if any(r.exit_code == 0 for r in runs):
            continue
        errors = sorted({r.error or f"exit code {r.exit_code}" for r in runs})
        out[case_name] = {"attempts": len(runs), "errors": errors}
    return out


def aggregate(records: list[RunRecord]) -> dict[str, dict]:
    """Per-case summary over the measured (non-warmup, successful) runs only.

    Reports median and full range rather than mean and standard deviation. With the repeat counts
    that are realistic here (3-5), a mean is easily dragged by one outlier -- and for the managed
    VLM backends an outlier is not noise but a real, recurring event (provider-side throttling,
    a slow region). The median plus min/max says what typically happens *and* how bad it gets,
    without implying a normal distribution nobody has shown exists.
    """
    by_case: dict[str, list[RunRecord]] = {}
    for record in records:
        if record.warmup or record.exit_code != 0:
            continue
        by_case.setdefault(record.case, []).append(record)

    summary: dict[str, dict] = {}
    for case_name, runs in by_case.items():
        walls = [r.wall_s for r in runs]
        entry: dict = {
            "n": len(runs),
            "wall_s_median": round(statistics.median(walls), 2),
            "wall_s_min": round(min(walls), 2),
            "wall_s_max": round(max(walls), 2),
            "note": runs[0].note,
            "units": runs[0].units,
        }
        # VRAM delta is only meaningful against a stable baseline, and on a shared GPU it often
        # is not. nvidia-smi reports whole-board usage, and per-process attribution requires
        # permissions this project's own dev machine does not grant ("[Insufficient
        # Permissions]"), so there is no way to isolate the pipeline's own allocation.
        #
        # Observed in a real sweep: the card sat at a steady 10,121 MiB of 12,282 held by
        # something outside the benchmark, and most runs measured a sane 685-1137 MiB delta on
        # top. But two runs happened to capture their baseline during a window when that external
        # allocation was absent (1,837 and 2,687 MiB), producing a "9,043 MiB" delta for a
        # configuration whose peak was identical to every other run's. Publishing that as a
        # measurement would be worse than publishing nothing.
        #
        # So: only runs whose baseline agrees with the case's median baseline contribute, and if
        # too few agree the figure is withheld rather than reported.
        baselines = [r.gpu.get("vram_baseline_mib") for r in runs if r.gpu.get("vram_baseline_mib") is not None]
        if baselines:
            median_baseline = statistics.median(baselines)
            entry["vram_baseline_mib_median"] = round(median_baseline, 1)
            entry["vram_baseline_spread_mib"] = round(max(baselines) - min(baselines), 1)
            stable = [
                r for r in runs
                if r.gpu.get("vram_baseline_mib") is not None
                and abs(r.gpu["vram_baseline_mib"] - median_baseline) <= BASELINE_TOLERANCE_MIB
            ]
        else:
            stable = []
        peaks = [r.gpu.get("vram_peak_delta_mib") for r in stable if r.gpu.get("vram_peak_delta_mib") is not None]
        if peaks and len(peaks) * 2 >= len(runs):
            entry["vram_peak_delta_mib_median"] = round(statistics.median(peaks), 1)
            entry["vram_delta_runs_used"] = len(peaks)
        else:
            entry["vram_peak_delta_mib_median"] = None
            entry["vram_unstable"] = True
        utils = [r.gpu.get("util_gpu_mean_pct") for r in runs if r.gpu.get("util_gpu_mean_pct") is not None]
        if utils:
            entry["util_gpu_mean_pct_median"] = round(statistics.median(utils), 1)
        energies = [r.gpu.get("energy_wh") for r in runs if r.gpu.get("energy_wh") is not None]
        if energies:
            entry["energy_wh_median"] = round(statistics.median(energies), 4)

        # Per-stage medians, keyed by stage name. A stage that ran more than once in a single run
        # (none do today, but pipeline.py is free to change that) is summed within that run first,
        # so the median is always "time this stage cost per run".
        stage_totals: dict[str, list[float]] = {}
        stage_models: dict[str, str | None] = {}
        for run in runs:
            per_run: dict[str, float] = {}
            for stage in run.stages:
                per_run[stage.stage] = per_run.get(stage.stage, 0.0) + stage.duration_s
                stage_models.setdefault(stage.stage, stage.model)
            for stage_name, total in per_run.items():
                stage_totals.setdefault(stage_name, []).append(total)
        entry["stages"] = {
            stage_name: {
                "model": stage_models.get(stage_name),
                "median_s": round(statistics.median(values), 2),
                "min_s": round(min(values), 2),
                "max_s": round(max(values), 2),
            }
            for stage_name, values in sorted(stage_totals.items())
        }
        summary[case_name] = entry
    return summary
