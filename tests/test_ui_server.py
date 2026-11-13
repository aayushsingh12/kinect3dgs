import http.client
import json
import queue
import re
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from helpers_ui import FakeStream
from kinect3dgs.dataset.synthetic import make_scene
from kinect3dgs.dataset.validate import validate_dataset
from kinect3dgs.ui.controller import Controller
from kinect3dgs.ui.server import UiServer


class Client:
    def __init__(self, server):
        self.s = server
        html = urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{server.port}/", headers={"Host": f"127.0.0.1:{server.port}"})).read().decode()
        self.t = re.search(r'const T = "([^"]+)"', html).group(1)

    def post(self, path, body=None):
        r = urllib.request.Request(f"http://127.0.0.1:{self.s.port}/api/{path}", data=json.dumps(body or {}).encode(), method="POST",
                                   headers={"X-Token": self.t, "Content-Type": "application/json", "Host": f"127.0.0.1:{self.s.port}"})
        try:
            return json.loads(urllib.request.urlopen(r, timeout=30).read())
        except urllib.error.HTTPError as e:
            return {**json.loads(e.read() or b"{}"), "http": e.code}

    def state(self):
        r = urllib.request.Request(f"http://127.0.0.1:{self.s.port}/api/state", headers={"X-Token": self.t, "Host": f"127.0.0.1:{self.s.port}"})
        return json.loads(urllib.request.urlopen(r).read())

    def wait(self, state, timeout=15):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self.state()["state"] == state:
                return
            time.sleep(0.1)
        raise AssertionError(f"state {self.state()['state']}, wanted {state}")


def make_ui(tmp_path, views=None):
    c = Controller(demo=True, sessions_dir=tmp_path / "sessions", datasets_dir=tmp_path / "datasets",
                   stream_factory=lambda: FakeStream(views=views, fps=100))
    s = UiServer(c, 0)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    c.start()
    cl = Client(s)
    cl.wait("ready")
    return c, s, cl


@pytest.fixture
def ui(tmp_path):
    c, s, cl = make_ui(tmp_path)
    yield c, s, cl
    c.shutdown()
    s.close()


def test_sse_state_and_metrics_and_video(ui):
    c, s, cl = ui
    conn = http.client.HTTPConnection("127.0.0.1", s.port, timeout=10)
    conn.request("GET", f"/events?t={cl.t}", headers={"Host": f"127.0.0.1:{s.port}"})
    resp = conn.getresponse()
    assert resp.status == 200 and resp.getheader("Content-Type") == "text/event-stream"
    first = resp.fp.readline().decode()
    assert first.startswith("event: state")
    assert cl.post("scan/start", {"name": "sse1", "plan": "table", "out": ""})["ok"]
    cl.wait("scanning")
    kinds = set()
    t0 = time.monotonic()
    while time.monotonic() - t0 < 5 and not {"metrics", "state"} <= kinds:
        line = resp.fp.readline().decode()
        if line.startswith("event:"):
            kinds.add(line.split(":", 1)[1].strip())
    assert {"metrics", "state"} <= kinds
    conn.close()
    v = http.client.HTTPConnection("127.0.0.1", s.port, timeout=10)
    v.request("GET", f"/video?t={cl.t}&view=rgb", headers={"Host": f"127.0.0.1:{s.port}"})
    r = v.getresponse()
    chunk = r.fp.read(4000)
    assert b"--frame" in chunk and b"image/jpeg" in chunk and b"\xff\xd8" in chunk
    v.close()
    v = http.client.HTTPConnection("127.0.0.1", s.port, timeout=10)
    v.request("GET", f"/video?t={cl.t}&view=depth", headers={"Host": f"127.0.0.1:{s.port}"})
    assert b"\xff\xd8" in v.getresponse().fp.read(4000)
    v.close()


def test_state_endpoint_and_validate_out(ui, tmp_path):
    c, s, cl = ui
    st = cl.state()
    assert st["state"] == "ready" and set(st["plans"]) == {"table", "room"} and st["demo"] is True
    assert cl.post("validate-out", {"out": "", "name": "x1", "plan": "table"})["ok"]
    r = cl.post("validate-out", {"out": "/", "name": "x1", "plan": "table"})
    assert not r["ok"] and "refusing" in r["error"]
    r = cl.post("validate-out", {"out": str(Path(__file__).parent / "t"), "name": "x1", "plan": "table"})
    assert not r["ok"] and "project folder" in r["error"]
    assert cl.post("scan/finish").get("http") == 409                                  # not scanning


def test_state_survives_tab_reload(ui):
    c, s, cl = ui
    cl.post("scan/start", {"name": "re1", "plan": "room", "out": ""})
    cl.wait("scanning")
    cl.post("scan/resume")
    time.sleep(0.5)
    other = Client(s)                                                                # "a new tab": new page load, new fetch
    st = other.state()
    assert st["state"] == "scanning" and st["scan"]["name"] == "re1" and st["scan"]["recording"] is True and st["plan"]["id"] == "room"


@pytest.mark.colmap
def test_build_through_the_ui_into_chosen_folder(tmp_path):
    scene = make_scene(24, 2.0)
    views = list(zip(scene.images, scene.depth_mm)) + [(scene.images[0], scene.depth_mm[0])]     # 25 views: all reached with every-3rd sampling
    c, s, cl = make_ui(tmp_path, views)
    events = c.hub.subscribe()
    try:
        out = tmp_path / "chosen_folder"
        assert cl.post("scan/start", {"name": "full", "plan": "table", "out": str(out)})["ok"]
        cl.wait("scanning")
        for k in range(3):
            cl.post("scan/resume")
            time.sleep(1.3)
            assert cl.post("scan/next", {})["ok"] if k < 2 else cl.post("scan/finish")["ok"]
        cl.wait("scan_done")
        assert cl.state()["scan"]["sessions"] == ["full_p1", "full_p2", "full_p3"]
        assert cl.post("process/start")["ok"]
        cl.wait("done", timeout=240)
        res = cl.state()["result"]
        assert res["ok"] and res["out"] == str(out) and res["registered"] >= 0.9 * res["selected"]
        assert validate_dataset(out).ok
        stages = []
        while True:
            try:
                kind, data = events.get_nowait()
            except queue.Empty:
                break
            if kind == "progress" and data["status"] == "start":
                stages.append(data["stage"])
        assert stages == ["select", "features", "matching", "mapping", "scale", "undistort", "export", "validate"]
        assert [p for p in tmp_path.iterdir() if ".tmp-" in p.name] == []
    finally:
        c.shutdown()
        s.close()


@pytest.mark.colmap
def test_cancel_build_leaves_nothing(tmp_path):
    scene = make_scene(24, 2.0)
    views = list(zip(scene.images, scene.depth_mm)) + [(scene.images[0], scene.depth_mm[0])]
    c, s, cl = make_ui(tmp_path, views)
    try:
        out = tmp_path / "cancelled"
        cl.post("scan/start", {"name": "can", "plan": "table", "out": str(out)})
        cl.wait("scanning")
        cl.post("scan/resume")
        time.sleep(1.5)
        cl.post("scan/finish")
        cl.wait("scan_done")
        assert cl.post("process/start")["ok"]
        time.sleep(1.5)
        assert cl.post("process/cancel")["ok"]
        cl.wait("scan_done", timeout=60)
        st = cl.state()
        assert st["error"] == "cancelled" and not out.exists()
        assert [p for p in tmp_path.iterdir() if ".tmp-" in p.name] == []
        assert cl.post("process/start")["ok"]                                         # can be started again after a cancel
        cl.post("process/cancel")
        cl.wait("scan_done", timeout=60)
    finally:
        c.shutdown()
        s.close()
