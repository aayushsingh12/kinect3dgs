"""AKS-002 hardware ladder R1-R8. Run: KINECT_HW=1 uv run pytest -m hardware tests/test_hardware_stream.py -s"""
import json
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from kinect3dgs.sessions import load_session
from kinect3dgs.stream import KinectStream

pytestmark = pytest.mark.hardware
ROOT = Path(__file__).resolve().parent.parent
ART = Path(__file__).parent / "_artifacts"
ART.mkdir(exist_ok=True)


def save(name, obj):
    (ART / f"{name}.json").write_text(json.dumps(obj, indent=2, default=str))
    print(f"\n[{name}]", json.dumps(obj, default=str))


def rec(tmp_path, *args, stdin=None):
    return subprocess.Popen([sys.executable, "-m", "kinect3dgs.record", "--out", str(tmp_path), *args],
                            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, stdin=stdin)


def shm_clean():
    return [f for f in os.listdir("/dev/shm") if f.startswith("kinect3dgs-")] == []


def test_r1_r3_stream_60s_rate_and_pairing():
    with KinectStream(lock_exposure=False) as ks:
        t_end = None
        while True:
            p = ks.get_pair(2.0)
            if p and t_end is None:
                t_end = time.monotonic() + 60
            if t_end and time.monotonic() >= t_end:
                break
        st = ks.stats
        sk = np.abs(st.skews_ms)
        out = {"first_pair_s": st.first_pair_s, "pairs": st.pairs, "mean_fps": st.mean_fps, "dropped_consumer": st.dropped_consumer,
               "bad_pairs": st.bad_pairs, "unmatched": st.unmatched, "skew_p50": float(np.percentile(sk, 50)),
               "skew_p95": float(np.percentile(sk, 95)), "skew_max": float(sk.max()), "restarts": st.restarts,
               "host_skew_p5": float(np.percentile(st.host_skews_ms, 5)), "host_skew_p95": float(np.percentile(st.host_skews_ms, 95))}
    save("r2_r3_stream", out)
    assert 28.0 <= out["mean_fps"] <= 31.5
    assert out["dropped_consumer"] == 0
    assert out["skew_max"] <= 16.7 and out["unmatched"] <= 0.02 * out["pairs"]   # RGB/depth sensors are not synchronised: bound is half a frame
    assert shm_clean()


def test_r4_record_30s(tmp_path):
    p = rec(tmp_path, "--seconds", "30", "--every", "3", "--exposure", "auto", "--name", "r4")
    out, _ = p.communicate(timeout=180)
    assert p.returncode == 0, out
    s = load_session(tmp_path / "r4")
    st = s.meta["stats"]
    save("r4_record", {**st, "bytes_per_pair": st["bytes_written"] / max(1, len(s))})
    assert 270 <= len(s) <= 310 and st["dropped_writer"] == 0 and st["write_errors"] == 0
    rgb, d, _ = s.frame(len(s) // 2)
    assert rgb.shape == (480, 640, 3) and (d > 0).mean() > 0.10
    assert shm_clean()


def test_r5_snapshot_10(tmp_path):
    p = rec(tmp_path, "--mode", "snapshot", "--exposure", "auto", "--name", "r5", stdin=subprocess.PIPE)
    time.sleep(14)                       # open + warm-up
    for _ in range(10):
        p.stdin.write("\n"); p.stdin.flush(); time.sleep(0.7)
    p.stdin.write("q\n"); p.stdin.flush()
    out, _ = p.communicate(timeout=60)
    assert p.returncode == 0, out
    s = load_session(tmp_path / "r5")
    assert len(s) == 10 and [r["idx"] for r in s.rows] == list(range(10))


def test_r6_shutdown_matrix(tmp_path):
    results = []
    cases = [("normal", ["--seconds", "3"], None), ("frames", ["--frames", "5"], None),
             ("sigint", ["--seconds", "60"], signal.SIGINT), ("sigterm", ["--seconds", "60"], signal.SIGTERM),
             ("normal2", ["--seconds", "3"], None)]
    for name, args, sig in cases:
        p = rec(tmp_path, *args, "--exposure", "auto", "--name", f"r6_{name}")
        if sig:
            time.sleep(16)
            p.send_signal(sig)
        out, _ = p.communicate(timeout=180)
        s = load_session(tmp_path / f"r6_{name}")
        st = s.meta["stats"]
        results.append({"case": name, "rc": p.returncode, "pairs": st["pairs_produced"], "restarts": st["restarts"],
                        "killed": st["producer_killed"], "first_pair_s": st["first_pair_s"], "shm_clean": shm_clean()})
        assert p.returncode in (0, 1) and st["pairs_produced"] > 0, out
        assert st["restarts"] == 0 and not st["producer_killed"] and shm_clean(), results[-1]
    save("r6_shutdown", results)


def test_r7_exposure_lock_applied(tmp_path):
    """The lock calls must succeed (they do). They are NOT effective on the 1473: `python -m kinect3dgs.exposure_check`
    showed the uncovered half still brightening +48 % with the lock on (+51 % auto). Hence default --exposure auto."""
    p = rec(tmp_path, "--seconds", "12", "--every", "10", "--exposure", "lock", "--name", "r7")
    out, _ = p.communicate(timeout=180)
    assert p.returncode == 0, out
    s = load_session(tmp_path / "r7")
    save("r7_exposure_flags", s.meta["exposure_lock"])
    assert s.meta["exposure_lock"]["applied"], s.meta["exposure_lock"]


def test_r8_resources(tmp_path):
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    p = rec(tmp_path, "--seconds", "20", "--exposure", "auto", "--name", "r8")
    p.communicate(timeout=120)
    peak_mb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    save("r8_resources", {"peak_rss_mb": peak_mb})
    assert peak_mb < 1500 and shm_clean()
