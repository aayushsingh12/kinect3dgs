"""End-to-end dataset build on a synthetic RGB-D scene with known poses (real pycolmap; ~1 min)."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pycolmap
import pytest

from kinect3dgs.dataset.__main__ import main as cli
from kinect3dgs.dataset.core import build_dataset, default_params
from kinect3dgs.dataset.synthetic import make_scene, write_session
from kinect3dgs.dataset.validate import validate_dataset

pytestmark = pytest.mark.colmap
SCALE_M = 2.0


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (xs ** 2).sum(1).mean()
    return s, R, mu_d - s * R @ mu_s


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn")
    scene = make_scene(24, SCALE_M)
    write_session(root / "sessions" / "syn", scene)
    events = []
    p = default_params([str(root / "sessions" / "syn")], out=str(root / "ds"), name="syn",
                       work_dir=str(root / "work"), target_frames=24, quiet=True)
    summary = build_dataset(p, on_progress=events.append)
    return root, scene, summary, events


def poses(path, names):
    rec = pycolmap.Reconstruction(Path(path) / "sparse" / "0")
    out = {}
    for im in rec.images.values():
        T = np.linalg.inv(np.vstack([im.cam_from_world().matrix(), [0, 0, 0, 1]]))
        out[im.name] = T
    return out


def test_registers_and_validates(built):
    root, scene, summary, _ = built
    assert summary["registered"] >= 0.9 * 24 and summary["points"] >= 3000 and summary["mean_error_px"] < 1.0
    rep = validate_dataset(root / "ds")
    assert rep.ok, rep.checks
    for f in ("dataset.json", "README.txt", "images", "sparse/0/cameras.bin"):
        assert (root / "ds" / f).exists()
    meta = json.loads((root / "ds" / "dataset.json").read_text())
    assert meta["sfm"]["registered"] == summary["registered"] and meta["camera"]["model"] == "PINHOLE"
    assert meta["selection"]["picks"][0]["session"] == "syn"


def test_pose_and_metric_scale_accuracy(built):
    root, scene, _, _ = built
    est = poses(root / "ds", None)
    names = [n for n in est]
    gt = {f"syn_{i:06d}.png": T for i, T in enumerate(scene.c2w)}
    common = [n for n in names if n in gt]
    assert len(common) >= 20
    s, R, t = umeyama(np.array([gt[n][:3, 3] for n in common]), np.array([est[n][:3, 3] for n in common]))
    assert abs(s - 1.0) < 0.03, f"metric scale off by {abs(s - 1):.1%}"           # D3: within 3 %
    errs_t, errs_r = [], []
    for n in common:
        c_gt = s * R @ gt[n][:3, 3] + t
        errs_t.append(np.linalg.norm(c_gt - est[n][:3, 3]))
        Rrel = est[n][:3, :3].T @ (R @ gt[n][:3, :3])
        errs_r.append(np.degrees(np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1))))
    assert np.median(errs_t) < 0.02 * SCALE_M and np.median(errs_r) < 1.0


def test_progress_events_ordered(built):
    _, _, _, events = built
    stages = []
    for e in events:
        if e["status"] == "start":
            stages.append(e["stage"])
    assert stages == ["select", "features", "matching", "mapping", "scale", "undistort", "export", "validate"]
    t = [e["elapsed_s"] for e in events]
    assert t == sorted(t)
    assert events[-1]["status"] == "done"


def test_no_metric_scale_flag(built, tmp_path):
    root, _, _, _ = built
    p = default_params([str(root / "sessions" / "syn")], out=str(tmp_path / "ds2"), name="syn2",
                       work_dir=str(tmp_path / "w2"), target_frames=24, quiet=True, metric_scale=False)
    build_dataset(p)
    meta = json.loads((tmp_path / "ds2" / "dataset.json").read_text())
    assert meta["metric_scale"]["applied"] is False


def test_force_replaces_dataset_and_refuses_otherwise(built, tmp_path):
    root, _, _, _ = built
    out = tmp_path / "ds"
    args = ["build", "--sessions", str(root / "sessions" / "syn"), "--out", str(out), "--work-dir", str(tmp_path / "w"),
            "--target-frames", "24", "--quiet"]
    assert cli(args) == 0
    assert cli(args) == 2                                  # non-empty, no --force
    (out / "marker.txt").write_text("old")
    assert cli(args + ["--force"]) == 0
    assert not (out / "marker.txt").exists()               # replaced, not merged
    assert [p for p in tmp_path.iterdir() if ".tmp-" in p.name or ".old-" in p.name] == []


def test_quality_failure_leaves_out_untouched(tmp_path):
    scene = make_scene(12, SCALE_M)
    scene.images = [np.full_like(im, 128) for im in scene.images]          # no texture at all
    write_session(tmp_path / "sessions" / "flat", scene)
    out = tmp_path / "ds"
    rc = cli(["build", "--sessions", str(tmp_path / "sessions" / "flat"), "--out", str(out),
              "--work-dir", str(tmp_path / "w"), "--target-frames", "12", "--min-sharpness-frac", "0", "--quiet"])
    assert rc == 3 and not out.exists()
    assert [p for p in tmp_path.iterdir() if ".tmp-" in p.name] == []


def test_sigterm_cancels_without_output(built, tmp_path):
    root, _, _, _ = built
    out = tmp_path / "ds"
    p = subprocess.Popen([sys.executable, "-m", "kinect3dgs.dataset", "build", "--sessions", str(root / "sessions" / "syn"),
                          "--out", str(out), "--work-dir", str(tmp_path / "w"), "--target-frames", "24", "--quiet"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(2.0)
    p.send_signal(signal.SIGTERM)
    p.communicate(timeout=120)
    assert p.returncode == 130
    assert not out.exists() and [q for q in tmp_path.iterdir() if ".tmp-" in q.name] == []


def test_jsonl_progress_cli(built, tmp_path, capsys):
    root, _, _, _ = built
    rc = cli(["build", "--sessions", str(root / "sessions" / "syn"), "--out", str(tmp_path / "ds"), "--work-dir",
              str(tmp_path / "w"), "--target-frames", "24", "--progress-jsonl"])
    assert rc == 0
    lines = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.strip().startswith("{")]
    assert lines[-1]["result"]["ok"] is True
    assert {l["stage"] for l in lines[:-1]} >= {"select", "features", "matching", "mapping", "scale", "undistort"}


def _copy(built, tmp_path):
    import shutil
    dst = tmp_path / "copy"
    shutil.copytree(built[0] / "ds", dst)
    return dst


def test_validator_detects_problems(built, tmp_path):
    d = _copy(built, tmp_path)
    assert validate_dataset(d).ok
    first = sorted((d / "images").glob("*.png"))[0]
    first.unlink()                                                    # missing image
    r = validate_dataset(d)
    assert not r.ok and any(n == "registered images present" and not p for n, p, _ in r.checks)
    d2 = tmp_path / "copy2"
    import shutil
    shutil.copytree(built[0] / "ds", d2)
    img = sorted((d2 / "images").glob("*.png"))[0]
    import cv2
    cv2.imwrite(str(img), cv2.resize(cv2.imread(str(img)), (320, 240)))      # wrong size
    r = validate_dataset(d2)
    assert not r.ok and any(n == "image sizes match camera" and not p for n, p, _ in r.checks)
    assert not validate_dataset(tmp_path / "nothing").ok                     # not a dataset at all
    assert not validate_dataset(d2, min_points=10**7).ok                      # too few points


def test_info_and_validate_cli(built, capsys):
    root = built[0]
    assert cli(["validate", str(root / "ds")]) == 0
    assert "VALID" in capsys.readouterr().out
    assert cli(["info", str(root / "ds")]) == 0
    assert "registered" in capsys.readouterr().out
    assert cli(["validate", str(root / "nope")]) == 1
