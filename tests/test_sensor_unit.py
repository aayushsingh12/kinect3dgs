import sys
import time
import types

import numpy as np
import pytest

from kinect3dgs import sensor
from kinect3dgs.errors import KinectBusyError, KinectNotFoundError, KinectTimeoutError


class FakeConn:
    def __init__(self):
        self.sent = []

    def send(self, m):
        self.sent.append(m)

    def close(self):
        pass


def test_worker_copies_buffers(monkeypatch):
    rgb = np.ones((480, 640, 3), np.uint8)
    depth = np.ones((480, 640), np.uint16)
    fake = types.SimpleNamespace(
        VIDEO_RGB=0, DEPTH_MM=5,
        sync_get_video=lambda format=0: (rgb, 1),
        sync_get_depth=lambda format=5: (depth, 2),
        sync_stop=lambda: None)
    monkeypatch.setitem(sys.modules, "freenect", fake)
    conn = FakeConn()
    sensor._worker(conn, 1, 0.0)
    status, out_rgb, out_depth, *_ = conn.sent[0]
    assert status == "ok"
    rgb[:] = 9
    depth[:] = 9
    assert out_rgb.max() == 1 and out_depth.max() == 1


def test_worker_none_means_open_error(monkeypatch):
    fake = types.SimpleNamespace(
        VIDEO_RGB=0, DEPTH_MM=5, sync_get_video=lambda format=0: None,
        sync_get_depth=lambda format=5: None, sync_stop=lambda: None)
    monkeypatch.setitem(sys.modules, "freenect", fake)
    conn = FakeConn()
    sensor._worker(conn, 1, 0.0)
    assert conn.sent == [("error", "open")]


def _err_target(conn, warmup, secs):
    conn.send(("error", "open"))


def _hang_target(conn, warmup, secs):
    time.sleep(60)


def test_error_maps_to_not_found(monkeypatch, tmp_path):
    monkeypatch.setattr(sensor, "PROC_MODULES", tmp_path / "none")
    with pytest.raises(KinectNotFoundError):
        sensor.capture(_target=_err_target, _check_devices=False, retries=0)


def test_error_maps_to_busy_when_gspca_loaded(monkeypatch, tmp_path):
    f = tmp_path / "modules"
    f.write_text("gspca_kinect 16384 0 - Live 0x0\n")
    monkeypatch.setattr(sensor, "PROC_MODULES", f)
    with pytest.raises(KinectBusyError):
        sensor.capture(_target=_err_target, _check_devices=False, retries=0)


def test_timeout_kills_child():
    t0 = time.monotonic()
    with pytest.raises(KinectTimeoutError):
        sensor.capture(fps_seconds=0.0, timeout=1.0, retries=0, _target=_hang_target, _check_devices=False)
    assert time.monotonic() - t0 < 15


def test_list_devices_fake_sysfs(monkeypatch, tmp_path):
    for name, vid, pid in [("1-2", "045e", "02ae"), ("1-3", "1d6b", "0002")]:
        d = tmp_path / name
        d.mkdir()
        (d / "idVendor").write_text(vid + "\n")
        (d / "idProduct").write_text(pid + "\n")
    monkeypatch.setattr(sensor, "SYSFS_USB", tmp_path)
    devs = sensor.list_devices()
    assert [d.pid for d in devs] == ["02ae"]


def test_retry_resets_camera_then_gives_up(monkeypatch, tmp_path):
    monkeypatch.setattr(sensor, "PROC_MODULES", tmp_path / "none")
    calls = []
    with pytest.raises(KinectNotFoundError):
        sensor.capture(retries=2, _target=_err_target, _check_devices=False, _reset=lambda: calls.append(1))
    assert len(calls) == 2


def test_busy_is_not_retried(monkeypatch, tmp_path):
    f = tmp_path / "modules"
    f.write_text("gspca_kinect 16384 0 - Live 0x0\n")
    monkeypatch.setattr(sensor, "PROC_MODULES", f)
    calls = []
    with pytest.raises(KinectBusyError):
        sensor.capture(retries=2, _target=_err_target, _check_devices=False, _reset=lambda: calls.append(1))
    assert calls == []
