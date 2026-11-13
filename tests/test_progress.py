import io
import json

import pytest

from kinect3dgs.dataset.errors import Cancelled
from kinect3dgs.dataset.progress import Emitter


def test_jsonl_schema_and_order():
    buf = io.StringIO()
    e = Emitter(jsonl=True, stream=buf)
    with e.stage("select"):
        pass
    with e.stage("features"):
        e.event("features", "progress", 0.5, "half")
    e.result({"ok": True})
    lines = [json.loads(l) for l in buf.getvalue().splitlines()]
    assert [l.get("stage") for l in lines[:5]] == ["select", "select", "features", "features", "features"]
    assert lines[-1] == {"result": {"ok": True}}
    for l in lines[:-1]:
        assert set(l) == {"stage", "status", "fraction", "message", "elapsed_s"}
    fr = [l["fraction"] for l in lines if l.get("stage") == "features" and l["fraction"] is not None]
    assert fr == sorted(fr)


def test_callback_and_cancel():
    seen = []
    e = Emitter(callback=seen.append, quiet=True)
    e.cancel.set()
    with pytest.raises(Cancelled):
        with e.stage("mapping"):
            pass
    assert seen == []


def test_error_event_on_exception():
    seen = []
    e = Emitter(callback=seen.append, quiet=True)
    with pytest.raises(ValueError):
        with e.stage("matching"):
            raise ValueError("boom")
    assert seen[-1]["status"] == "error" and seen[-1]["message"] == "boom"
