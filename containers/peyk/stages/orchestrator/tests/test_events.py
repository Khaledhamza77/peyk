import json
import sys
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import events


def _emit_and_capture(fn) -> str:
    """Runs fn() with stdout captured, returns exactly the one line events.emit() printed —
    fails loudly if fn() printed anything else or nothing at all, so a test can't silently pass
    on the wrong line."""
    buf = StringIO()
    real_stdout = sys.stdout
    sys.stdout = buf
    try:
        fn()
    finally:
        sys.stdout = real_stdout
    lines = [line for line in buf.getvalue().splitlines() if line]
    assert len(lines) == 1, f"expected exactly one printed line, got {lines!r}"
    return lines[0]


def test_emit_prints_one_line_prefixed_and_json_decodable():
    events.set_job_id(None)
    line = _emit_and_capture(lambda: events.emit("tsr", "dispatch_start", model="tableformer"))
    assert line.startswith(events.EVENT_PREFIX)
    record = json.loads(line[len(events.EVENT_PREFIX):])
    assert record["stage"] == "tsr"
    assert record["event"] == "dispatch_start"
    assert record["model"] == "tableformer"
    assert "ts" in record


def test_emit_includes_whatever_job_id_is_currently_set():
    events.set_job_id("job-123")
    try:
        line = _emit_and_capture(lambda: events.emit("ocr", "dispatch_end", exit_code=0))
        record = json.loads(line[len(events.EVENT_PREFIX):])
        assert record["job_id"] == "job-123"
    finally:
        events.set_job_id(None)


def test_emit_extra_fields_pass_through_untouched():
    line = _emit_and_capture(
        lambda: events.emit("ocr", "stub", label="text", doc_stem="cib_sample", region_id="r7")
    )
    record = json.loads(line[len(events.EVENT_PREFIX):])
    assert record["label"] == "text"
    assert record["doc_stem"] == "cib_sample"
    assert record["region_id"] == "r7"


def test_set_job_id_is_global_and_overwritable():
    events.set_job_id("first")
    events.set_job_id("second")
    line = _emit_and_capture(lambda: events.emit("layout", "dispatch_start"))
    record = json.loads(line[len(events.EVENT_PREFIX):])
    assert record["job_id"] == "second"
    events.set_job_id(None)
