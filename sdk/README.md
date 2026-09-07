# peyk

Python SDK to configure, orchestrate, and run the `peyk` document-parsing pipeline's Docker
containers (see the repo root [README](../README.md) for what the pipeline itself does).

Talks to Docker via [docker-py](https://docker-py.readthedocs.io/) — no shell subprocess, so none
of `containers/peyk/run_local.sh`'s MSYS/Windows path-rewriting workarounds are needed here.

## Install

The repo root is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) with
this package (`sdk/`) as its one member. From the repo root, run once:

```
uv sync
```

This creates `.venv` at the repo root and installs this package into it editable, plus its `dev`
extra (pytest).

`uv sync --extra notebook` additionally installs jupyter/ipykernel to run `demo.ipynb` (that extra
lives on the root [`pyproject.toml`](../pyproject.toml), not this package's — a downstream
`pip install peyk` consumer of this package on its own has no use for it).

See [`demo.py`](../demo.py) / [`demo.ipynb`](../demo.ipynb) at the repo root for a runnable
end-to-end example against `hotstorage/`.

## Usage

```python
from pathlib import Path
from peyk import Peyk, PipelineConfig, StageConfig

peyk = Peyk(image="peyk:dev")  # build it first via peyk.build_image("containers/peyk") if needed

peyk.configure(
    PipelineConfig(
        layout=StageConfig(model="heron"),
        tsr=StageConfig(model="surya"),
        ocr=StageConfig(model="paddleocr-vl", lang="arabic"),
        cell_ocr=StageConfig(model="surya"),
        figures=StageConfig(model="gemini-3-1-flash-lite"),
    ),
    config_dir="./peyk-config",
)

peyk.set_credentials(bedrock_bearer_token="...")  # only if a selected backend needs it

peyk.ensure_sidecars(wait=True)  # starts + waits only the sidecars this config actually needs

result = peyk.run(input_dir="./hotstorage/input", output_dir="./hotstorage/output")
print(result.exit_code, result.logs, result.job_id)

peyk.stop_sidecars()  # optional explicit teardown
```

## Job history & artifacts

Every `run()` call is recorded automatically in a local SQLite store (`peyk.jobs`, default
`~/.peyk/peyk.db`) — a job row plus one event per pipeline-stage dispatch (start/end, duration,
exit code, and any stub-fallback/credential errors), parsed out of the container's own log via
the `@@PEYK-EVENT@@` lines `containers/peyk/stages/orchestrator/events.py` emits. This happens
regardless of any other option below — it's cheap (text only) and always on.

```python
for job in peyk.jobs.list_jobs():
    print(job.job_id, job.status, job.exit_code)

for event in peyk.jobs.get_events(result.job_id):
    print(event.stage, event.event, event.duration_s, event.artifact_path)
```

Pass `persist_artifacts=True` to `run()` to additionally copy each dispatched stage's own
input/output directory (crops, per-region model JSON/HTML, viz PNGs) out of the shared
`peyk-hotstorage-workdir` volume into a local, stage-partitioned tree (`peyk.artifacts`, default
`~/.peyk/artifacts/<stage>/<job_id>/...`) — useful for tracing an actual model error back to the
exact crop/output that produced it. Off by default: a single job's table-cell OCR crops alone can
be hundreds of files.

```python
result = peyk.run(input_dir="...", output_dir="...", persist_artifacts=True)

# Each dispatch_end event's artifact_path now points at the copied directory/directories:
for event in peyk.jobs.get_events(result.job_id, stage="ocr"):
    print(event.artifact_path)

# Clean up later, independently of the DB history (which stays queryable either way):
peyk.artifacts.cleanup(job_id=result.job_id)   # this job, every stage
peyk.artifacts.cleanup(stage="ocr")            # every job's ocr artifacts
```

`Peyk(db_path=..., artifacts_root=...)` overrides both default locations.

## What's not covered yet

- No CLI — Python API only.
- `sidecars.py`/`runner.py`/`client.py` have no automated tests: they need a real Docker daemon +
  GPU to mean anything. Verify end-to-end manually against the repo's own `hotstorage/`/`data/`
  sample PDFs, the same way `containers/peyk/run_local.sh` is verified.
- `KNOWN_VLM_MODELS` in `config.py` is a static copy of `example.yaml`'s own model-tier list — it
  can drift behind the container's real registry (`docker run --rm peyk:dev --stage vlm
  --list-models`). Pass `allowed_vlm_models=...` to `PipelineConfig.validate()` if it has.

## Tests

```
.venv/Scripts/python.exe -m pytest sdk/tests/    # Windows
.venv/bin/python -m pytest sdk/tests/            # Linux/macOS
```

`test_config.py` (schema validation/YAML round-tripping) and `test_history.py` (JobStore/
ArtifactStore — event parsing, job/event queries, artifact cleanup) — neither needs Docker.
`sidecars.py`/`runner.py`/`client.py` still have no automated tests of their own, per the note
above.
