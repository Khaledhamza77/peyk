"""Extract a partial report from the JobStore while a sweep is still running, or after it was
interrupted -- without touching the live process at all.

Harness.run_matrix() only accumulates RunRecords in memory and calls save() once, at the very
end. If time runs out (or the process is killed) before that, nothing written this way is lost --
every job's per-stage timings were already persisted to the JobStore the moment the container
exited (PeykRunner.run() calls job_store.finish_job()/ingest_log() synchronously, per job, whether
or not the harness process that launched it ever gets to its own final save()). This module reads
that same database independently and rebuilds RunRecords good enough for report.py's existing
aggregate()/render() functions, so a mid-sweep snapshot uses the exact same reporting code path as
a completed run -- not a separate, less-trusted format.

Two things this reconstruction cannot recover, because they were never persisted anywhere except
the live harness process's own memory:
  - GPU/VRAM/utilization data (sampled by GpuSampler, held only in RunRecord.gpu).
  - Work-unit counts (WorkUnits.count(), held only in RunRecord.units).
A snapshot report is therefore latency-only -- no VRAM column, no s/page normalisation. Fine for
"what's the relative ranking so far", not a substitute for the full run's report.
"""
from __future__ import annotations

import time
from pathlib import Path

from peyk import JobStore

from . import cases as cases_module
from .harness import RunRecord, StageTiming


def _case_names_by_config_yaml() -> dict[str, list[str]]:
    """Maps each known case's exact `config.to_yaml()` text to every case name that produces it.

    A list, not a single name, because several cases are legitimately byte-identical: each sweep
    in cases.py is built by overriding one slot away from _base(), so "override a slot to its own
    default value" reproduces _base() exactly. Confirmed in practice -- layout=heron, ocr=paddleocr,
    tsr=tableformer, and force_scanned=on (born-digital's "on" state, already _base()'s default)
    are all the literal base config. A dict keyed by config text with a single value would let
    whichever case iterates last silently claim every completed job for that config, making the
    others look like they never ran even though they did. Every name sharing a config gets credit
    for the same underlying runs instead -- accurate, if slightly redundant to read.

    Reliable in the first place because config_yaml is the exact text PeykRunner.run() reads off
    disk and stores verbatim as JobStore's `config_yaml` column -- Peyk.configure() writes
    config.to_yaml() right before every run, so two jobs for the same case always match byte-for-byte.
    """
    lookup: dict[str, list[str]] = {}
    for sweep_name in cases_module.SWEEPS:
        for case in cases_module.build([sweep_name]):
            lookup.setdefault(case.config.to_yaml(), []).append(case.name)
    return lookup


def _stage_timings_for(job_store: JobStore, job_id: str) -> list[StageTiming]:
    """Same pairing logic as Harness._stage_timings(), minus the GPU-window slicing (no GpuTrace
    exists for a job reconstructed after the fact) -- every StageTiming.gpu is left empty."""
    events = job_store.get_events(job_id)
    pending: dict[str, list] = {}
    timings: list[StageTiming] = []
    for evt in events:
        if evt.event == "dispatch_start":
            pending.setdefault(evt.stage or "unknown", []).append(evt)
        elif evt.event == "dispatch_end":
            queue = pending.get(evt.stage or "unknown", [])
            start_evt = queue.pop(0) if queue else None
            duration = evt.duration_s if evt.duration_s is not None else 0.0
            started = start_evt.ts if start_evt is not None and start_evt.ts else (evt.ts or 0.0) - duration
            ended = evt.ts or (started + duration)
            timings.append(
                StageTiming(
                    stage=evt.stage or "unknown", model=evt.model, duration_s=duration,
                    started_at=started, ended_at=ended, exit_code=evt.exit_code, gpu={},
                )
            )
    return timings


def partial_records(since_ts: float, job_store: JobStore | None = None) -> list[RunRecord]:
    """Every job started at or after `since_ts` that matches a known benchmark case, reconstructed
    as RunRecords ready for harness.aggregate()/report.render_from_records()-style consumption.

    `since_ts` matters: without it, a job from an earlier session that happened to use an
    identical config (the same base layout/tsr/ocr defaults get reused across sweeps, and across
    separate runs of this harness over time) would be silently folded into the current snapshot.
    Pass the current run's own start time -- e.g. time.time() captured right before launching it,
    or the earliest `started_at` you know belongs to this run.

    Warmup detection: JobStore does not record which of a case's runs was the discarded warmup
    (that flag lives only on the harness's own in-memory RunRecord). Reconstructed instead from
    execution order, which is deterministic -- Harness.run_case() always dispatches the warmup
    first, then repeats in order -- so the earliest successful job per case is treated as warmup.
    """
    job_store = job_store or JobStore()
    by_config = _case_names_by_config_yaml()

    # ended_at is not None <=> the job actually finished (succeeded or failed) -- a still-running
    # job has status='running', exit_code=None, ended_at=None. Excluding it here, not just
    # defaulting its missing exit_code to something, matters concretely: treating a None exit_code
    # as failure fabricates a failed run for a case that (confirmed in practice) went on to
    # succeed -- the job simply had not finished yet at snapshot time. A still-running job also
    # has no place in the warmup-by-execution-order inference below; it is neither the warmup nor
    # a measured repeat until it actually completes.
    jobs = [
        j for j in job_store.list_jobs()
        if j.started_at and j.started_at >= since_ts and j.ended_at is not None
    ]
    by_case: dict[str, list] = {}
    for job in jobs:
        case_names = by_config.get(job.config_yaml or "")
        if not case_names:
            continue  # not one of ours (or config text didn't match byte-for-byte) -- skip rather than guess
        for case_name in case_names:
            by_case.setdefault(case_name, []).append(job)

    records: list[RunRecord] = []
    for case_name, case_jobs in by_case.items():
        case_jobs.sort(key=lambda j: j.started_at)
        for i, job in enumerate(case_jobs):
            wall_s = (job.ended_at - job.started_at) if job.ended_at else 0.0
            records.append(
                RunRecord(
                    case=case_name,
                    repeat=i - 1,  # -1 for the (assumed) warmup, 0.. for the rest -- cosmetic only
                    job_id=job.job_id,
                    exit_code=job.exit_code if job.exit_code is not None else 1,
                    wall_s=round(wall_s, 3),
                    warmup=(i == 0),
                    stages=_stage_timings_for(job_store, job.job_id) if job.status == "succeeded" else [],
                    gpu={"available": False, "reason": "not captured in a live/partial snapshot"},
                    units={},
                    note="partial snapshot -- GPU/VRAM and work-unit counts unavailable",
                )
            )
    return records


def snapshot_report(since_ts: float, job_store: JobStore | None = None) -> str:
    """Renders the same report.py sections a completed run would, from whatever has finished so
    far. Explicitly labelled as partial in its own header -- this must never be mistaken for a
    completed sweep's report."""
    from . import report as report_module

    job_store = job_store or JobStore()
    records = partial_records(since_ts, job_store)
    summary = report_module.aggregate(records)

    completed_cases = sorted(summary.keys())
    all_case_names = [c.name for n in cases_module.SWEEPS for c in cases_module.build([n])]

    header = (
        f"# peyk benchmark -- PARTIAL SNAPSHOT (sweep still in progress)\n\n"
        f"Captured {time.strftime('%Y-%m-%dT%H:%M:%S')}. {len(completed_cases)} of "
        f"{len(all_case_names)} planned configurations have at least one completed measured run.\n\n"
        "**No GPU/VRAM data in this snapshot** -- that is only ever computed by the live harness "
        "process and was not persisted anywhere it can be recovered from independently. Re-run "
        "`python -m benchmarks report` against the full result file once the sweep finishes for "
        "complete numbers.\n\n"
    )
    parts = [
        header,
        report_module.latency_table(summary),
        report_module.stage_breakdown_table(summary),
        report_module.failures_block(records),
    ]

    still_pending = [n for n in all_case_names if n not in completed_cases]
    if still_pending:
        parts.append(
            "## Not yet completed\n\n" + "\n".join(f"- {n}" for n in still_pending) + "\n"
        )
    return "\n".join(p for p in parts if p)
