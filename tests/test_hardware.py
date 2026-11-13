import pytest

from kinect3dgs import checks, sensor

pytestmark = pytest.mark.hardware


def test_device_listed():
    assert any(d.pid in sensor.CAMERA_PIDS for d in sensor.list_devices())


def test_capture_passes_all_checks():
    cap = sensor.capture(warmup=15, fps_seconds=3.0)
    for r in (checks.check_shapes(cap.rgb, cap.depth_mm), checks.check_rgb_content(cap.rgb),
              checks.check_depth_content(cap.depth_mm), checks.check_stream(cap.stamps, cap.seconds)):
        assert r.passed, f"{r.rung} {r.name}: {r.detail}"


def test_repeated_open_close():
    for _ in range(5):
        rgb, depth = sensor.grab_pair(warmup=2)
        assert rgb.shape == (480, 640, 3) and depth.shape == (480, 640)
