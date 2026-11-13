import os
import sys
import time
from pathlib import Path

import pytest

from kinect3dgs.errors import KinectNotFoundError, KinectTimeoutError
from kinect3dgs.stream import KinectStream

FAKE = str(Path(__file__).parent / "fake_freenect")


def _dev(pid):
    from kinect3dgs.sensor import UsbDevice
    return [UsbDevice("1-2", "045e", pid, "")]


@pytest.fixture
def fake(monkeypatch):
    assert "freenect" not in sys.modules
    from kinect3dgs import stream as stream_mod
    monkeypatch.setattr(stream_mod.sensor, "list_devices", lambda: _dev("02ae"))   # never read the real USB bus
    monkeypatch.syspath_prepend(FAKE)
    monkeypatch.setenv("FAKE_FPS", "100")
    monkeypatch.setenv("FAKE_SKEW_MS", "5")
    yield monkeypatch
    sys.modules.pop("freenect", None)


def shm_leftovers(name):
    return os.path.exists("/dev/shm/" + name)


def test_pairs_end_to_end_no_aliasing(fake):
    with KinectStream(open_timeout=30, _reset=lambda: None) as ks:
        name = ks.ring.name
        got = []
        t0 = time.monotonic()
        while len(got) < 100 and time.monotonic() - t0 < 30:
            p = ks.get_pair(1.0)
            if p:
                got.append(p)
        assert len(got) == 100
        for p in got:
            i = int(p.depth[0, 0]) - 1
            assert int(p.rgb[0, 0, 0]) == i % 256          # rgb and depth of the same frame index
            assert 4.0 < p.skew_ms < 6.0                    # injected 5 ms on the device clock
            assert (p.rgb[10:20, 10:20] == i % 256).all()   # buffer not aliased to a later frame
    assert not shm_leftovers(name)


def test_slow_consumer_drops_oldest(fake):
    with KinectStream(open_timeout=30, _reset=lambda: None) as ks:
        n = 0
        t0 = time.monotonic()
        while n < 10 and time.monotonic() - t0 < 30:
            p = ks.get_pair(1.0)
            if p:
                n += 1
                time.sleep(0.3)
        assert ks.stats.dropped_consumer > 0


def test_stop_is_fast_and_idempotent(fake):
    ks = KinectStream(open_timeout=30, _reset=lambda: None)
    ks.start()
    while ks.get_pair(1.0) is None:
        pass
    t0 = time.monotonic()
    ks.stop()
    ks.stop()
    assert time.monotonic() - t0 < 8 and not ks.killed


def test_stall_watchdog_restarts_then_raises(fake):
    fake.setenv("FAKE_MODE", "stall")
    resets = []
    ks = KinectStream(open_timeout=30, stall_timeout=1.0, max_restarts=1, _reset=lambda: resets.append(1))
    ks.start()
    with pytest.raises(KinectTimeoutError):
        t0 = time.monotonic()
        while time.monotonic() - t0 < 60:
            ks.get_pair(0.5)
    assert ks.stats.restarts == 1 and len(resets) == 1


def test_cannot_open_raises_not_found(fake):
    fake.setenv("FAKE_MODE", "noopen")
    ks = KinectStream(max_restarts=1, _reset=lambda: None)
    ks.start()
    with pytest.raises(KinectNotFoundError):
        t0 = time.monotonic()
        while time.monotonic() - t0 < 60:
            ks.get_pair(0.5)


def test_exposure_lock_results(fake):
    with KinectStream(lock_exposure=True, lock_after_s=0.2, _reset=lambda: None) as ks:
        t0 = time.monotonic()
        while ks.stats.lock_results is None and time.monotonic() - t0 < 30:
            ks.get_pair(0.5)
        assert ks.stats.lock_results == {"AUTO_EXPOSURE": 0, "AUTO_WHITE_BALANCE": 0, "AUTO_FLICKER": 0}


def test_exposure_lock_failure_reported(fake):
    fake.setenv("FAKE_FLAG_RET", "-1")
    with KinectStream(lock_exposure=True, lock_after_s=0.2, _reset=lambda: None) as ks:
        t0 = time.monotonic()
        while ks.stats.lock_results is None and time.monotonic() - t0 < 30:
            ks.get_pair(0.5)
        assert set(ks.stats.lock_results.values()) == {-1}


def test_disconnect_waits_and_resumes(fake, tmp_path):
    from kinect3dgs import stream as stream_mod
    fake.setenv("FAKE_MODE", "die_once")
    fake.setenv("FAKE_DIE_MARKER", str(tmp_path / "died"))
    gone = {"until": None}

    def list_devices():
        if (tmp_path / "died").exists() and gone["until"] is None:
            gone["until"] = time.monotonic() + 2.5            # unplugged for 2.5 s after the producer died
        absent = gone["until"] is not None and time.monotonic() < gone["until"]
        return [] if absent else _dev("02ae")

    fake.setattr(stream_mod.sensor, "list_devices", list_devices)
    ks = KinectStream(open_timeout=30, usb_wait=30, _reset=lambda: None)
    ks.start()
    seqs_after = 0
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < 60 and seqs_after < 20:
            p = ks.get_pair(0.5)
            if p and any(e["event"] == "reconnected" for e in ks.stats.events):
                seqs_after += 1
    finally:
        names = [e["event"] for e in ks.stats.events]
        ks.stop()
    assert "disconnected" in names and "reconnected" in names and seqs_after >= 20
    assert ks.stats.restarts == 0


def test_disconnect_never_returns_raises(fake, tmp_path):
    from kinect3dgs import stream as stream_mod
    fake.setenv("FAKE_MODE", "die_once")
    fake.setenv("FAKE_DIE_MARKER", str(tmp_path / "died"))
    fake.setattr(stream_mod.sensor, "list_devices",
                 lambda: [] if (tmp_path / "died").exists() else _dev("02ae"))
    ks = KinectStream(open_timeout=30, usb_wait=2.0, _reset=lambda: None)
    ks.start()
    with pytest.raises(KinectNotFoundError):
        t0 = time.monotonic()
        while time.monotonic() - t0 < 60:
            ks.get_pair(0.5)
