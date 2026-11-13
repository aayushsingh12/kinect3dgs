import json

import numpy as np
import pytest

from kinect3dgs.sessions import SCHEMA_VERSION, SessionError, load_session, write_json_atomic
from kinect3dgs.writer import PairWriter


def make_session(tmp_path, n=5):
    meta = {"schema_version": SCHEMA_VERSION, "started_utc": "x", "formats": {}, "args": {}, "stats": {}, "intrinsics": None}
    write_json_atomic(tmp_path / "session.json", meta)
    w = PairWriter(tmp_path)
    rng = np.random.default_rng(1)
    data = []
    for i in range(n):
        rgb = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        d = rng.integers(0, 5000, (480, 640), dtype=np.uint16)
        row = {"idx": i, "t_host_s": i * 0.1, "ts_rgb": i, "ts_depth": i, "skew_ms": 1.0, "sharpness": 1.0,
               "depth_valid_frac": 0.5, "bad_pair": False}
        assert w.submit(i, rgb, d, row)
        data.append((rgb, d))
    w.close()
    return data


def test_roundtrip_equality(tmp_path):
    data = make_session(tmp_path)
    s = load_session(tmp_path)
    assert len(s) == 5 and s.meta["intrinsics"] is None
    for (rgb, d), (r2, d2, row) in zip(data, s.frames()):
        assert np.array_equal(rgb, r2) and np.array_equal(d, d2)


def test_missing_session_json(tmp_path):
    with pytest.raises(SessionError):
        load_session(tmp_path)


def test_wrong_schema(tmp_path):
    (tmp_path / "session.json").write_text(json.dumps({"schema_version": 99}))
    with pytest.raises(SessionError):
        load_session(tmp_path)


def test_missing_image_file(tmp_path):
    make_session(tmp_path, 3)
    (tmp_path / "depth" / "000001.png").unlink()
    with pytest.raises(SessionError):
        load_session(tmp_path)


def test_noncontiguous_idx(tmp_path):
    make_session(tmp_path, 3)
    lines = (tmp_path / "frames.jsonl").read_text().splitlines()
    (tmp_path / "frames.jsonl").write_text("\n".join([lines[0], lines[2]]) + "\n")
    with pytest.raises(SessionError):
        load_session(tmp_path)
