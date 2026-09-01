"""Local job/step traceability — ingests the @@PEYK-EVENT@@ JSON lines emitted by
containers/peyk/stages/orchestrator/events.py out of a job's combined container log into a small
SQLite database (JobStore), and optionally copies each dispatched stage's own input/output
directory (crops, per-region model JSON, viz PNGs) out of the shared peyk-hotstorage-workdir
named volume into a stage-partitioned local directory tree (ArtifactStore) for offline debugging.

Two separate concerns, two separate knobs, wired together in runner.py:
- The events/job/step history is always recorded (cheap, text-only) — every PeykRunner.run()
  call gets a queryable trace regardless of persist_artifacts.
- Artifact copying is opt-in per call (persist_artifacts=True) since a single job's table-cell
  crops alone can be hundreds of files — see PeykRunner.run()'s own docstring.

ArtifactStore's layout is stage-first: <root>/<stage>/<job_id>/<relative-workdir-path>/... — not
job-first — so "clean up every ocr artifact across every job" and "clean up everything from job
X" are both a single directory-tree walk, and so debugging a recurring per-stage failure means
browsing one directory rather than reassembling it from many jobs' folders.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

EVENT_PREFIX = "@@PEYK-EVENT@@"

DEFAULT_DB_PATH = Path.home() / ".peyk" / "peyk.db"
DEFAULT_ARTIFACTS_ROOT = Path.home() / ".peyk" / "artifacts"

# workdir's container-side mount point (runner.py mounts the same fixed path — see
# CONFIG_CONTAINER_DIR/its "/hotstorage/workdir" literal) — used to turn an event's
# absolute-in-container input_dir/output_dir into a path relative to the workdir volume's root,
# which is what's actually reachable from the alpine helper container ArtifactStore's copy uses.
WORKDIR_CONTAINER_PREFIX = "/hotstorage/workdir/"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    config_yaml TEXT,
    input_dir TEXT,
    output_dir TEXT,
    started_at REAL,
    ended_at REAL,
    status TEXT,
    exit_code INTEGER
);
CREATE TABLE IF NOT EXISTS events (
    job_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    ts REAL,
    stage TEXT,
    event TEXT,
    model TEXT,
    duration_s REAL,
    exit_code INTEGER,
    doc_stem TEXT,
    region_id TEXT,
    message TEXT,
    input_dir TEXT,
    output_dir TEXT,
    artifact_path TEXT,
    extra_json TEXT,
    PRIMARY KEY (job_id, seq)
);
"""

_EVENT_COLUMNS = (
    "job_id", "seq", "ts", "stage", "event", "model", "duration_s", "exit_code",
    "doc_stem", "region_id", "message", "input_dir", "output_dir", "artifact_path", "extra_json",
)
# Top-level fields events.emit() always sends as named JSON keys — anything else in a parsed
# event line is stage-specific extra data, stashed in extra_json instead of a dedicated column.
_KNOWN_EVENT_FIELDS = {"ts", "job_id", "stage", "event", "model", "duration_s", "exit_code", "doc_stem", "region_id", "message", "input_dir", "output_dir"}


def parse_event_line(line: str) -> dict | None:
    """Finds and decodes one @@PEYK-EVENT@@ JSON payload out of a raw container log line, or
    None if this line isn't one (the common case — most lines are the existing human-readable
    "[peyk-orchestrator] ..." prints, untouched by this addition). Substring search rather than
    requiring the prefix at line-start: docker's combined stdout/stderr stream can interleave
    partial writes from concurrent output in principle, though in practice each print() call is
    one line/one write."""
    idx = line.find(EVENT_PREFIX)
    if idx == -1:
        return None
    try:
        return json.loads(line[idx + len(EVENT_PREFIX):])
    except (json.JSONDecodeError, ValueError):
        return None


def relative_to_workdir(container_path: str | None) -> str | None:
    """Strips the fixed /hotstorage/workdir/ container prefix off an event's input_dir/output_dir,
    returning None if it isn't under there (defensive — every real dispatch always is, since
    every stage's in/out dirs are created under the --workdir root; see pipeline.py)."""
    if not container_path:
        return None
    normalized = container_path.replace("\\", "/")
    if not normalized.startswith(WORKDIR_CONTAINER_PREFIX):
        return None
    return normalized[len(WORKDIR_CONTAINER_PREFIX):].rstrip("/") or None


@dataclass
class JobRecord:
    job_id: str
    config_yaml: str | None
    input_dir: str | None
    output_dir: str | None
    started_at: float | None
    ended_at: float | None
    status: str | None
    exit_code: int | None


@dataclass
class EventRecord:
    job_id: str
    seq: int
    ts: float | None
    stage: str | None
    event: str | None
    model: str | None
    duration_s: float | None
    exit_code: int | None
    doc_stem: str | None
    region_id: str | None
    message: str | None
    input_dir: str | None
    output_dir: str | None
    artifact_path: str | None
    extra: dict = field(default_factory=dict)


class JobStore:
    """One SQLite file, default ~/.peyk/peyk.db. Every PeykRunner.run() call creates a job row up
    front and ingests every @@PEYK-EVENT@@ line the container produced into the events table,
    tagged with the job_id the SDK itself assigned (not whatever job_id the container's own
    --job-id happened to be given — see runner.py) — that's the sole source of truth for which
    job an event belongs to, so ingestion never depends on trusting container-reported identity."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def create_job(self, job_id: str, config_yaml: str, input_dir: str, output_dir: str) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO jobs (job_id, config_yaml, input_dir, output_dir, started_at, status) "
                "VALUES (?, ?, ?, ?, ?, 'running')",
                (job_id, config_yaml, input_dir, output_dir, time.time()),
            )
            conn.commit()

    def finish_job(self, job_id: str, exit_code: int) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "UPDATE jobs SET ended_at = ?, status = ?, exit_code = ? WHERE job_id = ?",
                (time.time(), "succeeded" if exit_code == 0 else "failed", exit_code, job_id),
            )
            conn.commit()

    def record_event(self, job_id: str, record: dict) -> int:
        """Inserts one parsed event dict (from parse_event_line) under job_id, returning the
        row's own per-job sequence number (0, 1, 2, ... in ingestion order) — used later to
        address this exact row again once artifact persistence resolves its input_dir/output_dir
        to a host path (see set_artifact_path)."""
        extra = {k: v for k, v in record.items() if k not in _KNOWN_EVENT_FIELDS}
        with closing(self._connect()) as conn:
            seq = conn.execute("SELECT COALESCE(MAX(seq), -1) + 1 FROM events WHERE job_id = ?", (job_id,)).fetchone()[0]
            conn.execute(
                f"INSERT INTO events ({', '.join(_EVENT_COLUMNS)}) VALUES ({', '.join('?' * len(_EVENT_COLUMNS))})",
                (
                    job_id, seq, record.get("ts", time.time()), record.get("stage"), record.get("event"),
                    record.get("model"), record.get("duration_s"), record.get("exit_code"),
                    record.get("doc_stem"), record.get("region_id"), record.get("message"),
                    record.get("input_dir"), record.get("output_dir"), None,
                    json.dumps(extra) if extra else None,
                ),
            )
            conn.commit()
            return seq

    def ingest_log(self, job_id: str, logs: str) -> int:
        """Scans a job's full combined log text line by line and records every event line found.
        Returns how many were ingested. Safe to call with the whole blocking-collected log
        (PeykRunner.run()'s stream_logs=False path) or incrementally per streamed chunk."""
        count = 0
        for line in logs.splitlines():
            record = parse_event_line(line)
            if record is not None:
                self.record_event(job_id, record)
                count += 1
        return count

    def set_artifact_path(self, job_id: str, seq: int, artifact_path: str) -> None:
        with closing(self._connect()) as conn:
            conn.execute("UPDATE events SET artifact_path = ? WHERE job_id = ? AND seq = ?", (artifact_path, job_id, seq))
            conn.commit()

    def list_jobs(self, status: str | None = None) -> list[JobRecord]:
        query = "SELECT job_id, config_yaml, input_dir, output_dir, started_at, ended_at, status, exit_code FROM jobs"
        params: tuple = ()
        if status is not None:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY started_at DESC"
        with closing(self._connect()) as conn:
            rows = conn.execute(query, params).fetchall()
        return [JobRecord(*row) for row in rows]

    def get_job(self, job_id: str) -> JobRecord | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT job_id, config_yaml, input_dir, output_dir, started_at, ended_at, status, exit_code "
                "FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return JobRecord(*row) if row else None

    def get_events(self, job_id: str, stage: str | None = None, event: str | None = None) -> list[EventRecord]:
        query = (
            "SELECT job_id, seq, ts, stage, event, model, duration_s, exit_code, doc_stem, "
            "region_id, message, input_dir, output_dir, artifact_path, extra_json FROM events WHERE job_id = ?"
        )
        params: list = [job_id]
        if stage is not None:
            query += " AND stage = ?"
            params.append(stage)
        if event is not None:
            query += " AND event = ?"
            params.append(event)
        query += " ORDER BY seq ASC"
        with closing(self._connect()) as conn:
            rows = conn.execute(query, params).fetchall()
        return [EventRecord(*row[:-1], extra=json.loads(row[-1]) if row[-1] else {}) for row in rows]

    def delete_job(self, job_id: str) -> None:
        """Removes a job's DB rows (jobs + events) only — does not touch any files an
        ArtifactStore may have persisted for it; pair with ArtifactStore.cleanup(job_id=...) for
        that (Peyk.cleanup_artifacts() does both together)."""
        with closing(self._connect()) as conn:
            conn.execute("DELETE FROM events WHERE job_id = ?", (job_id,))
            conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            conn.commit()


class ArtifactStore:
    """Stage-partitioned local artifact tree: <root>/<stage>/<job_id>/<relative-workdir-path>/...
    PeykRunner._persist_artifacts populates this straight from the input_dir/output_dir every
    dispatch_end event already recorded — no separate hardcoded list of workdir subdirectory
    names (tsr_in, ocr_out, ...) to keep in sync with pipeline.py as it evolves."""

    def __init__(self, root: str | Path = DEFAULT_ARTIFACTS_ROOT):
        self.root = Path(root)

    def stage_job_dir(self, stage: str, job_id: str) -> Path:
        return self.root / stage / job_id

    def cleanup(self, job_id: str | None = None, stage: str | None = None) -> list[Path]:
        """Deletes matching artifact directories and returns the ones actually removed.
        job_id/stage are independent optional filters: stage alone clears that stage across every
        job, job_id alone clears that job's artifacts under every stage, both narrows to one
        directory, neither wipes the whole artifact root — all four are legitimate, deliberate
        calls, not guarded further here (Peyk.cleanup_artifacts() is the caller-facing entry
        point; it's on the caller to pass the filters they mean)."""
        removed: list[Path] = []
        if not self.root.exists():
            return removed
        stage_dirs = [self.root / stage] if stage is not None else [d for d in self.root.iterdir() if d.is_dir()]
        for stage_dir in stage_dirs:
            if not stage_dir.is_dir():
                continue
            job_dirs = [stage_dir / job_id] if job_id is not None else list(stage_dir.iterdir())
            for job_dir in job_dirs:
                if job_dir.is_dir():
                    shutil.rmtree(job_dir)
                    removed.append(job_dir)
            if not any(stage_dir.iterdir()):
                stage_dir.rmdir()
        return removed
