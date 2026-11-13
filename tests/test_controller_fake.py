import time

import pytest

from helpers_ui import FakeStream
from kinect3dgs.ui.controller import Controller
from kinect3dgs.sessions import load_session


@pytest.fixture
def ctrl(tmp_path, monkeypatch):
    holder = {}

    def factory():
        holder["stream"] = FakeStream(fps=100, **holder.get("kw", {}))
        return holder["stream"]

    c = Controller(demo=True, sessions_dir=tmp_path / "sessions", datasets_dir=tmp_path / "datasets", stream_factory=factory)
    c.holder = holder
    c.start()
    for _ in range(100):
        if c.state == "ready":
            break
        time.sleep(0.05)
    yield c
    c.shutdown()


def wait_state(c, state, timeout=10):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if c.state == state:
            return
        time.sleep(0.05)
    raise AssertionError(f"state {c.state}, wanted {state}")


def start(c, name="t1", plan="table"):
    ok, err = c.submit("start_scan", name=name, plan=plan, out="")
    assert ok, err
    wait_state(c, "scanning")


def test_full_flow_three_passes(ctrl, tmp_path):
    c = ctrl
    start(c)
    assert c.submit("resume")[0]
    time.sleep(1.2)
    assert c.scan["frames"] >= 20
    assert c.submit("next_pass")[0] and c.scan["pass_idx"] == 1
    c.submit("resume"); time.sleep(0.9)
    assert c.submit("next_pass")[0] and c.scan["pass_idx"] == 2
    c.submit("resume"); time.sleep(0.9)
    ok, err = c.submit("next_pass")
    assert not ok and "last pass" in err
    assert c.submit("finish")[0]
    wait_state(c, "scan_done")
    assert c.scan["sessions"] == ["t1_p1", "t1_p2", "t1_p3"]
    for n in c.scan["sessions"]:
        s = load_session(tmp_path / "sessions" / n)
        assert len(s) >= 15 and s.meta["args"]["plan"] == "table"
    assert c.holder["stream"].stopped


def test_invalid_transitions_and_names(ctrl):
    c = ctrl
    assert not c.submit("resume")[0] and not c.submit("process")[0] and not c.submit("finish")[0]
    for bad in ("", "A B", "../x", "x" * 41, "UPPER"):
        ok, err = c.submit("start_scan", name=bad, plan="table", out="")
        assert not ok and "name" in err
    ok, err = c.submit("start_scan", name="ok1", plan="nope", out="")
    assert not ok


def test_existing_session_name_refused(ctrl, tmp_path):
    (tmp_path / "sessions" / "dup_p1").mkdir(parents=True)
    ok, err = ctrl.submit("start_scan", name="dup", plan="table", out="")
    assert not ok and "already exists" in err


def test_short_pass_guard_and_force(ctrl):
    c = ctrl
    start(c)
    c.submit("resume"); time.sleep(0.3)
    ok, err = c.submit("next_pass")
    assert not ok and "press n again" in err
    assert c.submit("next_pass", force=True)[0] and c.scan["pass_idx"] == 1


def test_empty_pass_leaves_no_session(ctrl, tmp_path):
    c = ctrl
    start(c, "empty")
    assert c.submit("finish")[0]
    wait_state(c, "ready")
    assert c.error == "nothing was recorded"
    assert not list((tmp_path / "sessions").glob("empty_p*"))


def test_shutdown_finalises_open_session(ctrl, tmp_path):
    c = ctrl
    start(c, "abort")
    c.submit("resume"); time.sleep(1.0)
    c.shutdown()
    s = load_session(tmp_path / "sessions" / "abort_p1")
    assert s.meta["aborted"] == "signal" and len(s) > 5


def test_hint_appears_for_too_close(tmp_path):
    holder = {"kw": {"depth_mm": 450}}
    c = Controller(demo=True, sessions_dir=tmp_path / "s", datasets_dir=tmp_path / "d",
                   stream_factory=lambda: FakeStream(fps=100, depth_mm=450))
    c.start()
    try:
        for _ in range(100):
            if c.state == "ready":
                break
            time.sleep(0.05)
        assert c.submit("start_scan", name="h1", plan="table", out="")[0]
        wait_state(c, "scanning")
        time.sleep(1.5)
        assert c.metrics["hint"] == "too close" and c.metrics["distance_m"] == 0.45
    finally:
        c.shutdown()


def test_disconnect_banner_and_clear(tmp_path):
    fs = FakeStream(fps=100, events=[{"t": 1.0, "event": "disconnected", "reason": "x"}])
    c = Controller(demo=True, sessions_dir=tmp_path / "s", datasets_dir=tmp_path / "d", stream_factory=lambda: fs)
    c.start()
    try:
        for _ in range(100):
            if c.state == "ready":
                break
            time.sleep(0.05)
        c.submit("start_scan", name="b1", plan="table", out="")
        wait_state(c, "scanning")
        time.sleep(0.4)
        assert "disconnected" in c.banner
        fs.stats.events.append({"t": 5.0, "event": "reconnected"})
        time.sleep(0.4)
        assert c.banner == ""
    finally:
        c.shutdown()


def test_stream_failure_finalises_and_reports(tmp_path):
    fs = FakeStream(fps=100, fail_after=60)
    c = Controller(demo=True, sessions_dir=tmp_path / "s", datasets_dir=tmp_path / "d", stream_factory=lambda: fs)
    c.start()
    try:
        for _ in range(100):
            if c.state == "ready":
                break
            time.sleep(0.05)
        c.submit("start_scan", name="f1", plan="table", out="")
        wait_state(c, "scanning")
        c.submit("resume")
        t0 = time.monotonic()
        while c.state == "scanning" and time.monotonic() - t0 < 10:
            time.sleep(0.05)
        assert c.state in ("scan_done", "failed") and "camera" in c.error
        if c.state == "scan_done":
            s = load_session(tmp_path / "s" / "f1_p1")
            assert s.meta["aborted"] == "stalled"
    finally:
        c.shutdown()


def test_validate_out_rules(ctrl):
    ok, err = ctrl.validate_out("", "good-name", "table")
    assert ok
    ok, err = ctrl.validate_out("/", "good-name", "table")
    assert not ok and "refusing" in err
    from kinect3dgs.dataset.params import REPO
    ok, err = ctrl.validate_out(str(REPO / "src" / "x"), "good-name", "table")
    assert not ok and "project folder" in err
    ok, err = ctrl.validate_out("", "Bad Name", "table")
    assert not ok and "name" in err
