import numpy as np

from kinect3dgs import checks as c


def rgb_ok():
    rng = np.random.default_rng(0)
    return rng.integers(20, 200, (480, 640, 3), dtype=np.uint8)


def depth_ok(frac=0.95):
    d = np.full((480, 640), 1500, np.uint16)
    d[: int(480 * (1 - frac))] = 0
    return d


def test_shapes():
    assert c.check_shapes(rgb_ok(), depth_ok()).passed
    assert not c.check_shapes(rgb_ok()[:, :, :1], depth_ok()).passed
    assert not c.check_shapes(rgb_ok(), depth_ok().astype(np.uint8)).passed


def test_rgb_content():
    assert c.check_rgb_content(rgb_ok()).passed
    assert not c.check_rgb_content(np.full((480, 640, 3), 128, np.uint8)).passed
    assert not c.check_rgb_content(np.zeros((480, 640, 3), np.uint8)).passed


def test_depth_content():
    assert c.check_depth_content(depth_ok(0.95)).passed
    assert not c.check_depth_content(np.zeros((480, 640), np.uint16)).passed
    assert not c.check_depth_content(depth_ok(0.05)).passed
    far = np.full((480, 640), 9000, np.uint16)
    assert not c.check_depth_content(far).passed
    wall = np.full((480, 640), 5000, np.uint16)   # real 1473 capture had a wall at 4-5 m
    r = c.check_depth_content(wall)
    assert r.passed and "beyond 4 m" in r.detail


def test_stream_thresholds():
    ts = [i * 1000 for i in range(60)]
    assert c.check_stream(ts, 3.0).passed          # 20.0 fps
    assert not c.check_stream(ts[:59], 3.0).passed  # 19.67 fps
    assert not c.check_stream([5, 4, 6, 7] * 20, 3.0).passed


def test_stream_wraps():
    ts = [(2**32 - 2000 + i * 1000) % 2**32 for i in range(61)]
    assert c.check_stream(ts, 3.0).passed


def test_roundtrip(tmp_path):
    import cv2
    rgb, d = rgb_ok(), depth_ok()
    cv2.imwrite(str(tmp_path / "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(tmp_path / "depth_mm.png"), d)
    np.save(tmp_path / "depth_mm.npy", d)
    assert c.check_roundtrip(tmp_path, rgb, d).passed
    np.save(tmp_path / "depth_mm.npy", d + 1)
    assert not c.check_roundtrip(tmp_path, rgb, d).passed
