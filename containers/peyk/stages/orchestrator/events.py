"""Structured, machine-parseable traceability events — emitted *alongside* the existing
human-readable stderr prints in stages.py/pipeline.py, not instead of them. Each event is one
line of EVENT_PREFIX + JSON printed to stdout. sdk/src/peyk/history.py scans a job's combined
container log for these lines and ingests them into a local JobStore, keyed by the job_id the
SDK itself assigned (see runner.py) — the job_id set here is carried along for human-readable
context (e.g. run_local.sh's own terminal output) but the SDK's own tagging is authoritative.

Printed to stdout specifically, not stderr where every existing diagnostic print already goes:
keeps the two streams distinguishable in principle, though PeykRunner.run() currently reads them
combined (docker-py's default) — parse_event_line() finds EVENT_PREFIX by substring search
regardless of what else shares the line, so interleaving from either stream is harmless.
"""
import json
import time

EVENT_PREFIX = "@@PEYK-EVENT@@"

_job_id: str | None = None


def set_job_id(job_id: str) -> None:
    global _job_id
    _job_id = job_id


def emit(stage: str, event: str, **fields) -> None:
    """stage: the logical pipeline stage this event is about ("layout", "tsr", "table_full",
    "ocr", "cell_ocr", "figures", "dcr", "fullpage", "vlm") — independent of which backend
    container-internally implements it (e.g. tsr's `surya` backend still emits stage="tsr").
    event: "dispatch_start" | "dispatch_end" | "stub" | "error". fields: event-specific extra
    data — dispatch_start/dispatch_end carry model/input_dir/output_dir/duration_s/exit_code (the
    input_dir/output_dir are what sdk/src/peyk/history.py's artifact persistence later copies out
    of the shared workdir volume); stub carries doc_stem/region_id/label; error carries
    model/message."""
    record = {"ts": time.time(), "job_id": _job_id, "stage": stage, "event": event, **fields}
    print(EVENT_PREFIX + json.dumps(record), flush=True)
