import numpy as np

from helpers_ui import FakeStream
from kinect3dgs.recorder import SessionRecorder
from kinect3dgs.sessions import load_session


def test_recorder_standalone_roundtrip(tmp_path):
    fs = FakeStream(fps=1000)
    rec = SessionRecorder(tmp_path / "s", {"x": 1}, stream=fs)
    t0 = None
    for i in range(30):
        p = fs.get_pair(1.0)
        rec.see(p)
        if i % 3 == 0:
            import time
            now = time.monotonic()
            t0 = t0 or now
            assert rec.record(p, t0, now) is not None
    meta = rec.close(fs)
    s = load_session(tmp_path / "s")
    assert len(s) == 10 and meta["stats"]["pairs_recorded"] == 10 and meta["stats"]["pairs_produced"] == 30
    assert meta["schema_version"] == 1 and meta["intrinsics"] is None and meta["aborted"] is None
    rgb, depth, row = s.frame(3)
    assert np.array_equal(rgb, fs.views[9 % len(fs.views)][0]) and row["idx"] == 3


def test_recorder_deltas_per_pass(tmp_path):
    fs = FakeStream(fps=1000)
    fs.stats.dropped_consumer = 7                       # counted before this session began
    rec = SessionRecorder(tmp_path / "s", {}, stream=fs)
    fs.stats.dropped_consumer = 10
    for _ in range(5):
        rec.see(fs.get_pair(1.0))
    meta = rec.close(fs)
    assert meta["stats"]["dropped_consumer"] == 3        # only what happened during this session


def test_close_hard_and_aborted_flag(tmp_path):
    fs = FakeStream(fps=1000)
    rec = SessionRecorder(tmp_path / "s", {}, stream=fs)
    meta = rec.close(fs, aborted="signal", hard=True)
    assert meta["aborted"] == "signal"
    assert load_session(tmp_path / "s").meta["aborted"] == "signal"
