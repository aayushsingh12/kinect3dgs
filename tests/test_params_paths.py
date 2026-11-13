import json
from pathlib import Path

import pytest

from kinect3dgs.dataset import params as P
from kinect3dgs.dataset.errors import UsageError


def make(tmp_path, out="out", **kw):
    sess = tmp_path / "sess"
    sess.mkdir(exist_ok=True)
    (sess / "session.json").write_text("{}")
    return P.BuildParams(sessions=[sess], out=tmp_path / out, name="x", work_dir=tmp_path / "work", **kw)


def test_ok_new_folder(tmp_path):
    P.check_paths(make(tmp_path))


def test_nonempty_requires_force(tmp_path):
    p = make(tmp_path)
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "f.txt").write_text("x")
    with pytest.raises(UsageError, match="not empty"):
        P.check_paths(p)


def test_force_only_replaces_datasets(tmp_path):
    p = make(tmp_path, force=True)
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "f.txt").write_text("x")
    with pytest.raises(UsageError, match="not a dataset"):
        P.check_paths(p)
    (tmp_path / "out" / "dataset.json").write_text(json.dumps({}))
    P.check_paths(p)


def test_refuses_project_source_folders(tmp_path):
    for d in ("src", "tests", "scripts", "specs"):
        p = make(tmp_path)
        p.out = P.REPO / d / "ds"
        with pytest.raises(UsageError, match="project folder"):
            P.check_paths(p)


def test_refuses_home_root_and_session_overlap(tmp_path):
    for bad in (Path("/"), Path.home()):
        p = make(tmp_path)
        p.out = bad
        with pytest.raises(UsageError):
            P.check_paths(p)
    p = make(tmp_path)
    p.out = p.sessions[0]
    with pytest.raises(UsageError, match="overlaps"):
        P.check_paths(p)
    p = make(tmp_path)
    p.out = p.sessions[0] / "ds"
    with pytest.raises(UsageError, match="overlaps"):
        P.check_paths(p)


def test_missing_parent_and_work_inside_out(tmp_path):
    p = make(tmp_path)
    p.out = tmp_path / "nope" / "ds"
    with pytest.raises(UsageError, match="parent"):
        P.check_paths(p)
    p = make(tmp_path)
    p.work_dir = p.out / "work"
    with pytest.raises(UsageError, match="contain"):
        P.check_paths(p)


def test_resolve_session(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "SESSIONS_DIR", tmp_path)
    (tmp_path / "s1").mkdir()
    (tmp_path / "s1" / "session.json").write_text("{}")
    assert P.resolve_session("s1") == (tmp_path / "s1").resolve()
    with pytest.raises(UsageError, match="unknown session"):
        P.resolve_session("nope")
