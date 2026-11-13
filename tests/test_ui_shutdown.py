import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from kinect3dgs.sessions import load_session

ROOT = Path(__file__).resolve().parent.parent


def launch(tmp_path):
    env = {**os.environ, "KINECT3DGS_DEMO_VIEWS": "6"}
    p = subprocess.Popen([sys.executable, "-u", "-m", "kinect3dgs.ui", "--demo", "--port", "0"], cwd=ROOT, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    line = p.stdout.readline()
    m = re.search(r"http://127.0.0.1:(\d+)/", line)
    assert m, line
    port = int(m.group(1))
    html = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode()
    token = re.search(r'const T = "([^"]+)"', html).group(1)

    def post(path, body=None):
        r = urllib.request.Request(f"http://127.0.0.1:{port}/api/{path}", data=json.dumps(body or {}).encode(), method="POST",
                                   headers={"X-Token": token, "Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(r, timeout=30).read())

    def state():
        r = urllib.request.Request(f"http://127.0.0.1:{port}/api/state", headers={"X-Token": token})
        return json.loads(urllib.request.urlopen(r).read())
    return p, post, state


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_signal_during_scan_finalises_session(tmp_path, sig):
    name = f"shut{int(sig)}x{os.getpid()}"
    p, post, state = launch(tmp_path)
    sess = ROOT / "data" / "sessions" / f"{name}_p1"
    try:
        for _ in range(60):
            if state()["state"] == "ready":
                break
            time.sleep(0.2)
        assert post("scan/start", {"name": name, "plan": "table", "out": str(tmp_path / "ds")})["ok"]
        for _ in range(100):
            if state()["state"] == "scanning":
                break
            time.sleep(0.2)
        post("scan/resume")
        time.sleep(2.0)
        p.send_signal(sig)
        assert p.wait(timeout=20) == 0
        s = load_session(sess)
        assert s.meta["aborted"] == "signal" and len(s) > 5
        assert [f for f in os.listdir("/dev/shm") if f.startswith("kinect3dgs-")] == []
    finally:
        if p.poll() is None:
            p.kill()
        import shutil
        shutil.rmtree(sess, ignore_errors=True)
