import json

from peyk.history import ArtifactStore, EVENT_PREFIX, JobStore, parse_event_line, relative_to_workdir


def test_parse_event_line_finds_json_payload_by_prefix():
    line = EVENT_PREFIX + json.dumps({"stage": "tsr", "event": "dispatch_start"})
    assert parse_event_line(line) == {"stage": "tsr", "event": "dispatch_start"}


def test_parse_event_line_ignores_ordinary_log_lines():
    assert parse_event_line("[peyk-orchestrator] dispatching --stage tsr ...") is None


def test_parse_event_line_ignores_malformed_payload():
    assert parse_event_line(EVENT_PREFIX + "{not json") is None


def test_relative_to_workdir_strips_prefix():
    assert relative_to_workdir("/hotstorage/workdir/tsr_out") == "tsr_out"
    assert relative_to_workdir("/hotstorage/workdir/crops/cib_sample") == "crops/cib_sample"


def test_relative_to_workdir_rejects_paths_outside_workdir():
    assert relative_to_workdir("/hotstorage/input") is None
    assert relative_to_workdir(None) is None


def test_job_lifecycle_and_event_ingestion(tmp_path):
    store = JobStore(tmp_path / "peyk.db")
    store.create_job("job-1", config_yaml="layout:\n  model: heron\n", input_dir="/in", output_dir="/out")

    logs = "\n".join([
        "[peyk-orchestrator] job_id=job-1",
        EVENT_PREFIX + json.dumps({
            "stage": "tsr", "event": "dispatch_start", "model": "tableformer",
            "input_dir": "/hotstorage/workdir/tsr_in", "output_dir": "/hotstorage/workdir/tsr_out",
        }),
        "[peyk-orchestrator] --stage tsr: 4.21s",
        EVENT_PREFIX + json.dumps({
            "stage": "tsr", "event": "dispatch_end", "model": "tableformer", "duration_s": 4.21,
            "exit_code": 0, "input_dir": "/hotstorage/workdir/tsr_in", "output_dir": "/hotstorage/workdir/tsr_out",
        }),
        EVENT_PREFIX + json.dumps({
            "stage": "ocr", "event": "stub", "label": "text", "doc_stem": "cib_sample", "region_id": "r7",
        }),
    ])
    ingested = store.ingest_log("job-1", logs)
    assert ingested == 3

    store.finish_job("job-1", exit_code=0)

    job = store.get_job("job-1")
    assert job.status == "succeeded"
    assert job.exit_code == 0

    events = store.get_events("job-1")
    assert [e.event for e in events] == ["dispatch_start", "dispatch_end", "stub"]
    assert events[0].seq == 0 and events[2].seq == 2

    dispatch_end = store.get_events("job-1", stage="tsr", event="dispatch_end")
    assert len(dispatch_end) == 1
    assert dispatch_end[0].duration_s == 4.21
    assert dispatch_end[0].output_dir == "/hotstorage/workdir/tsr_out"

    stub_events = store.get_events("job-1", event="stub")
    assert stub_events[0].doc_stem == "cib_sample"
    assert stub_events[0].region_id == "r7"
    assert stub_events[0].extra == {"label": "text"}


def test_finish_job_failed_status_on_nonzero_exit(tmp_path):
    store = JobStore(tmp_path / "peyk.db")
    store.create_job("job-2", config_yaml="", input_dir="/in", output_dir="/out")
    store.finish_job("job-2", exit_code=1)
    assert store.get_job("job-2").status == "failed"


def test_list_jobs_filters_by_status_and_orders_newest_first(tmp_path):
    store = JobStore(tmp_path / "peyk.db")
    store.create_job("job-a", config_yaml="", input_dir="/in", output_dir="/out")
    store.finish_job("job-a", exit_code=0)
    store.create_job("job-b", config_yaml="", input_dir="/in", output_dir="/out")
    store.finish_job("job-b", exit_code=1)

    all_jobs = store.list_jobs()
    assert [j.job_id for j in all_jobs] == ["job-b", "job-a"]

    failed_only = store.list_jobs(status="failed")
    assert [j.job_id for j in failed_only] == ["job-b"]


def test_set_artifact_path_updates_only_the_addressed_row(tmp_path):
    store = JobStore(tmp_path / "peyk.db")
    store.create_job("job-1", config_yaml="", input_dir="/in", output_dir="/out")
    seq = store.record_event("job-1", {"stage": "tsr", "event": "dispatch_end", "input_dir": "/hotstorage/workdir/tsr_in"})
    other_seq = store.record_event("job-1", {"stage": "ocr", "event": "dispatch_end"})

    store.set_artifact_path("job-1", seq, "/home/user/.peyk/artifacts/tsr/job-1/tsr_in")

    events = {e.seq: e for e in store.get_events("job-1")}
    assert events[seq].artifact_path == "/home/user/.peyk/artifacts/tsr/job-1/tsr_in"
    assert events[other_seq].artifact_path is None


def test_delete_job_removes_jobs_and_events(tmp_path):
    store = JobStore(tmp_path / "peyk.db")
    store.create_job("job-1", config_yaml="", input_dir="/in", output_dir="/out")
    store.record_event("job-1", {"stage": "layout", "event": "dispatch_start"})

    store.delete_job("job-1")

    assert store.get_job("job-1") is None
    assert store.get_events("job-1") == []


def test_artifact_store_cleanup_by_job_and_stage(tmp_path):
    root = tmp_path / "artifacts"
    (root / "tsr" / "job-1").mkdir(parents=True)
    (root / "tsr" / "job-1" / "crop.png").write_bytes(b"x")
    (root / "tsr" / "job-2").mkdir(parents=True)
    (root / "ocr" / "job-1").mkdir(parents=True)
    store = ArtifactStore(root)

    removed = store.cleanup(job_id="job-1", stage="tsr")
    assert removed == [root / "tsr" / "job-1"]
    assert not (root / "tsr" / "job-1").exists()
    assert (root / "tsr" / "job-2").exists()
    assert (root / "ocr" / "job-1").exists()


def test_artifact_store_cleanup_whole_stage_across_jobs(tmp_path):
    root = tmp_path / "artifacts"
    (root / "tsr" / "job-1").mkdir(parents=True)
    (root / "tsr" / "job-2").mkdir(parents=True)
    (root / "ocr" / "job-1").mkdir(parents=True)
    store = ArtifactStore(root)

    removed = store.cleanup(stage="tsr")
    assert sorted(removed) == sorted([root / "tsr" / "job-1", root / "tsr" / "job-2"])
    assert not (root / "tsr").exists()
    assert (root / "ocr" / "job-1").exists()


def test_artifact_store_cleanup_one_job_across_stages(tmp_path):
    root = tmp_path / "artifacts"
    (root / "tsr" / "job-1").mkdir(parents=True)
    (root / "ocr" / "job-1").mkdir(parents=True)
    (root / "ocr" / "job-2").mkdir(parents=True)
    store = ArtifactStore(root)

    removed = store.cleanup(job_id="job-1")
    assert sorted(removed) == sorted([root / "tsr" / "job-1", root / "ocr" / "job-1"])
    assert not (root / "tsr").exists()  # emptied stage dir removed too
    assert (root / "ocr" / "job-2").exists()


def test_artifact_store_cleanup_on_missing_root_is_a_noop(tmp_path):
    store = ArtifactStore(tmp_path / "does-not-exist")
    assert store.cleanup() == []
