import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from kinect3dgs import record
from kinect3dgs.sessions import load_session

ROOT = Path(__file__).resolve().parent.parent
FAKE = str(Path(__file__).parent / "fake_freenect")


@pytest.fixture
def env(tmp_path):
    usb = tmp_path / "usb" / "1-2"
    usb.mkdir(parents=True)
    (usb / "idVendor").write_text("045e\n")
    (usb / "idProduct").write_text("02ae\n")
    e = {**os.environ, "PYTHONPATH": f"{FAKE}:{ROOT / 'src'}", "KINECT3DGS_SYSFS_USB": str(tmp_path / "usb"),
         "FAKE_FPS": "100", "FAKE_SKEW_MS": "5"}
    return e


def run(env, tmp_path, *args, **kw):
    return subprocess.Popen([sys.executable, "-m", "kinect3dgs.record", "--out", str(tmp_path / "out"),
                             "--exposure", "auto", "--warmup", "0.2", *args],
                            env=env, cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kw)


def test_continuous_seconds_and_every(env, tmp_path):
    p = run(env, tmp_path, "--seconds", "2", "--every", "3", "--name", "a")
    out, _ = p.communicate(timeout=60)
    assert p.returncode == 0, out
    s = load_session(tmp_path / "out" / "a")
    st = s.meta["stats"]
    assert abs(len(s) - st["pairs_produced"] / 3) <= 3 and len(s) > 20
    assert [r["idx"] for r in s.rows] == list(range(len(s)))
    assert s.meta["finished_utc"] and st["dropped_writer"] == 0 and st["mean_fps"] > 30
    assert s.meta["formats"]["depth"]["mode"] == "DEPTH_REGISTERED"


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_signals_leave_valid_session(env, tmp_path, sig):
    p = run(env, tmp_path, "--name", "b", "--seconds", "30")
    time.sleep(4)
    p.send_signal(sig)
    out, _ = p.communicate(timeout=30)
    assert p.returncode in (0, 1), out
    s = load_session(tmp_path / "out" / "b")
    assert s.meta["finished_utc"] and s.meta["aborted"] in ("signal", None) and len(s) > 0
    assert [f for f in os.listdir("/dev/shm") if f.startswith(f"kinect3dgs-{p.pid}-")] == []


def test_snapshot_enter_presses(env, tmp_path):
    p = run(env, tmp_path, "--mode", "snapshot", "--name", "c", stdin=subprocess.PIPE)
    time.sleep(3)
    for _ in range(4):
        p.stdin.write("\n")
        p.stdin.flush()
        time.sleep(0.6)
    p.stdin.write("q\n")
    p.stdin.flush()
    out, _ = p.communicate(timeout=30)
    assert p.returncode == 0, out
    s = load_session(tmp_path / "out" / "c")
    assert len(s) == 4 and out.count("saved") == 4


def test_refuses_existing_session(env, tmp_path):
    (tmp_path / "out" / "d").mkdir(parents=True)
    p = run(env, tmp_path, "--name", "d", "--seconds", "1")
    out, _ = p.communicate(timeout=30)
    assert p.returncode == 2 and "refusing" in out


def test_no_camera_exits_2(env, tmp_path):
    (tmp_path / "usb" / "1-2" / "idProduct").write_text("02c2\n")
    p = run(env, tmp_path, "--name", "e", "--seconds", "1")
    out, _ = p.communicate(timeout=30)
    assert p.returncode == 2 and "No Kinect camera" in out


def test_disk_low_refusal(monkeypatch, tmp_path):
    monkeypatch.setattr(record, "free_bytes", lambda p: 1)
    assert record.main(["--out", str(tmp_path), "--name", "x", "--seconds", "1"]) == 2
    assert not (tmp_path / "x").exists()


def test_preview_binds_loopback_only():
    from kinect3dgs.preview import HOST, Preview
    assert HOST == "127.0.0.1"
    pv = Preview(0)
    url = pv.start()
    try:
        assert pv.server.server_address[0] == "127.0.0.1" and url.startswith("http://127.0.0.1:")
        import urllib.request
        assert b"kinect3dgs preview" in urllib.request.urlopen(url, timeout=5).read()
    finally:
        pv.stop()
