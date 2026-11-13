"""State machine and worker thread for the guided UI (plan 3.1/3.2). HTTP threads only submit commands and read snapshots."""
from __future__ import annotations

import queue
import re
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .. import sensor
from ..dataset.errors import UsageError
from ..dataset.params import REPO, BuildParams, check_paths
from ..errors import KinectError
from ..recorder import SessionRecorder
from .coach import Coach
from .plans import PLANS, get_plan
from .process import ProcessManager

SESSIONS = REPO / "data" / "sessions"
DATASETS = REPO / "data" / "datasets"
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
MIN_PASS_FRAMES = 20
EVERY = 3

# command -> states in which it is accepted
ALLOWED = {
    "check": {"checking", "ready", "failed", "done"},
    "start_scan": {"ready"},
    "resume": {"scanning"}, "pause": {"scanning"}, "next_pass": {"scanning"}, "finish": {"scanning", "opening_camera"},
    "process": {"scan_done"}, "cancel": {"processing"}, "reset": {"scan_done", "done", "failed", "ready"},
}


class EventHub:
    def __init__(self):
        self.clients: list[queue.Queue] = []
        self.lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q = queue.Queue(maxsize=200)
        with self.lock:
            self.clients.append(q)
        return q

    def unsubscribe(self, q) -> None:
        with self.lock:
            if q in self.clients:
                self.clients.remove(q)

    def publish(self, kind: str, data: dict) -> None:
        with self.lock:
            for q in self.clients:
                try:
                    q.put_nowait((kind, data))
                except queue.Full:
                    try:
                        q.get_nowait()
                        q.put_nowait((kind, data))
                    except queue.Empty:
                        pass


class Command:
    def __init__(self, _name: str, **kw):
        self.name, self.kw = _name, kw
        self.done = threading.Event()
        self.ok, self.error = True, ""


class Controller:
    def __init__(self, demo: bool = False, sessions_dir: Path = SESSIONS, datasets_dir: Path = DATASETS,
                 stream_factory=None):
        self.demo = demo
        self.sessions_dir, self.datasets_dir = Path(sessions_dir), Path(datasets_dir)
        self.stream_factory = stream_factory
        self.hub = EventHub()
        self.cmds: queue.Queue[Command] = queue.Queue()
        self.lock = threading.RLock()
        self.stop_evt = threading.Event()
        self.state = "checking"
        self.checks: list[dict] = []
        self.banner = ""
        self.scan = {}          # name, plan, out, pass_idx, sessions[], recording, frames, rec_s
        self.metrics: dict = {}
        self.latest = None      # (rgb, depth)
        self.result: dict | None = None
        self.error = ""
        self.stream = self.recorder = self.coach = None
        self.proc = ProcessManager(self.hub.publish, self._build_done)
        self.last_out: Path | None = None
        self._last_metrics_t = 0.0
        self._ev_seen = 0
        self._rec_since = None
        self._n_pairs_pass = 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    # ---------------------------------------------------------------- public API (HTTP threads)
    def start(self) -> None:
        self.thread.start()

    def snapshot(self) -> dict:
        with self.lock:
            sc = dict(self.scan)
            plan = get_plan(sc["plan"]).public() if sc.get("plan") else None
            return {"state": self.state, "checks": self.checks, "banner": self.banner, "scan": sc, "plan": plan,
                    "plans": {k: v.public() for k, v in PLANS.items()}, "result": self.result, "error": self.error,
                    "metrics": self.metrics, "demo": self.demo, "default_name": datetime.now().strftime("scan-%Y%m%d-%H%M")}

    def submit(self, _name: str, _timeout: float = 15.0, **kw) -> tuple[bool, str]:
        with self.lock:
            if self.state not in ALLOWED.get(_name, set()):
                return False, f"'{_name}' is not possible while {self.state}"
        cmd = Command(_name, **kw)
        self.cmds.put(cmd)
        if not cmd.done.wait(_timeout):
            return False, "timed out"
        return cmd.ok, cmd.error

    def validate_out(self, out: str, name: str, plan: str) -> tuple[bool, str]:
        try:
            p = self._build_params(name, plan, out)
            if not out:
                self.datasets_dir.mkdir(parents=True, exist_ok=True)      # the default location may create itself
            check_paths(p)
            return True, ""
        except UsageError as e:
            return False, str(e)
        except KeyError as e:
            return False, str(e)

    def latest_frame(self):
        with self.lock:
            return self.latest

    def shutdown(self) -> None:
        self.stop_evt.set()
        self.thread.join(5.0)
        self._finalise(aborted="signal")
        self.proc.cancel()

    # ---------------------------------------------------------------- worker thread
    def _run(self) -> None:
        self._do_checks()
        while not self.stop_evt.is_set():
            self._drain()
            if self.state in ("opening_camera", "scanning") and self.stream is not None:
                try:
                    pair = self.stream.get_pair(0.1)
                except KinectError as e:
                    self._fail(f"camera: {e}", aborted="stalled")
                    continue
                if pair is not None:
                    self._on_pair(pair)
                self._banner_from_events()
            else:
                time.sleep(0.05)
            self._publish_metrics()

    def _drain(self) -> None:
        while True:
            try:
                cmd = self.cmds.get_nowait()
            except queue.Empty:
                return
            try:
                if self.state not in ALLOWED.get(cmd.name, set()):
                    raise UsageError(f"'{cmd.name}' is not possible while {self.state}")
                getattr(self, "_cmd_" + cmd.name)(**cmd.kw)
            except (UsageError, KeyError, ValueError) as e:
                cmd.ok, cmd.error = False, str(e)
            except Exception as e:  # noqa: BLE001  (surface unexpected errors to the UI, keep the worker alive)
                cmd.ok, cmd.error = False, f"{type(e).__name__}: {e}"
            cmd.done.set()

    # ---------------------------------------------------------------- commands
    def _set_state(self, state: str) -> None:
        with self.lock:
            self.state = state
        self.hub.publish("state", self.snapshot())

    def _do_checks(self) -> None:
        checks = []
        if self.demo:
            checks.append({"name": "kinect", "ok": True, "detail": "demo (synthetic camera)"})
        else:
            cam = [d for d in sensor.list_devices() if d.pid in sensor.CAMERA_PIDS]
            checks.append({"name": "kinect", "ok": bool(cam),
                           "detail": f"{cam[0].vid}:{cam[0].pid}" if cam else "not found - plug in the AC adapter and the USB cable"})
        free = shutil.disk_usage(REPO).free
        checks.append({"name": "disk", "ok": free > 3 * 1024**3, "detail": f"{free / 1e9:.0f} GB free"})
        try:
            import pycolmap  # noqa: F401
            checks.append({"name": "colmap", "ok": True, "detail": f"pycolmap {pycolmap.__version__}"})
        except Exception as e:  # noqa: BLE001
            checks.append({"name": "colmap", "ok": False, "detail": str(e)})
        with self.lock:
            self.checks = checks
            self.error = ""
            self.state = "ready" if all(c["ok"] for c in checks) else "checking"
        self.hub.publish("state", self.snapshot())

    def _cmd_check(self) -> None:
        self._do_checks()

    def _build_params(self, name: str, plan: str, out: str) -> BuildParams:
        if not NAME_RE.match(name or ""):
            raise UsageError("name: use 1-40 characters from a-z 0-9 _ -, starting with a letter or digit")
        get_plan(plan)
        sessions = [self.sessions_dir / f"{name}_p{i + 1}" for i in range(len(get_plan(plan).passes))]
        out_p = Path(out) if out else self.datasets_dir / name
        return BuildParams(sessions=sessions, out=out_p, name=name, work_dir=REPO / "data" / "work" / name)

    def _cmd_start_scan(self, name: str, plan: str, out: str = "") -> None:
        p = self._build_params(name, plan, out)
        if out == "":
            self.datasets_dir.mkdir(parents=True, exist_ok=True)
        for s in p.sessions:
            if s.exists():
                raise UsageError(f"session {s.name} already exists; choose another name")
        check_paths(p)
        factory = self.stream_factory
        if factory is None:
            if self.demo:
                from .demo import DemoStream
                factory = DemoStream
            else:
                from ..stream import KinectStream
                factory = KinectStream
        self.stream = factory()
        self.stream.start()
        with self.lock:
            self.scan = {"name": name, "plan": plan, "out": str(p.out), "pass_idx": 0, "sessions": [],
                         "recording": False, "frames": 0, "rec_s": 0.0}
            self.error, self.result, self._ev_seen, self.banner = "", None, 0, ""
        self._set_state("opening_camera")

    def _begin_pass(self) -> None:
        sc = self.scan
        k = sc["pass_idx"]
        plan = get_plan(sc["plan"])
        sess = self.sessions_dir / f"{sc['name']}_p{k + 1}"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.recorder = SessionRecorder(sess, {"ui": True, "plan": sc["plan"], "pass": k + 1, "every": EVERY},
                                        devices=[d for d in sensor.list_devices()] if not self.demo else [],
                                        stream=self.stream)
        if self.coach is None:
            self.coach = Coach(plan.passes[k])
        else:
            self.coach.set_pass(plan.passes[k])
        with self.lock:
            sc["frames"], sc["rec_s"], sc["recording"] = 0, 0.0, False
        self._n_pairs_pass, self._rec_since = 0, None

    def _cmd_resume(self) -> None:
        with self.lock:
            self.scan["recording"] = True
        self._rec_since = time.monotonic()
        self.hub.publish("state", self.snapshot())

    def _cmd_pause(self) -> None:
        self._accumulate_rec()
        with self.lock:
            self.scan["recording"] = False
        self.hub.publish("state", self.snapshot())

    def _accumulate_rec(self) -> None:
        if self._rec_since is not None:
            with self.lock:
                self.scan["rec_s"] += time.monotonic() - self._rec_since
            self._rec_since = None

    def _close_pass(self, aborted=None) -> None:
        if self.recorder is None:
            return
        self._accumulate_rec()
        rec, self.recorder = self.recorder, None
        meta = rec.close(self.stream, aborted)
        n = meta["stats"]["pairs_recorded"]
        if n == 0:
            shutil.rmtree(rec.root, ignore_errors=True)          # nothing recorded: leave no empty session
        else:
            with self.lock:
                self.scan["sessions"].append(rec.root.name)

    def _cmd_next_pass(self, force: bool = False) -> None:
        plan = get_plan(self.scan["plan"])
        if self.scan["pass_idx"] + 1 >= len(plan.passes):
            raise UsageError("this was the last pass; use finish")
        if self.scan["frames"] < MIN_PASS_FRAMES and not force:
            raise UsageError(f"only {self.scan['frames']} frames in this pass (need {MIN_PASS_FRAMES}); press n again to continue anyway")
        self._close_pass()
        with self.lock:
            self.scan["pass_idx"] += 1
        self._begin_pass()
        self.hub.publish("state", self.snapshot())

    def _cmd_finish(self) -> None:
        self._close_pass()
        if self.stream is not None:
            self.stream.stop()
            self.stream = None
        self.coach = None
        with self.lock:
            self.latest = None
            self.scan["recording"] = False
        self._set_state("scan_done" if self.scan["sessions"] else "ready")
        if not self.scan["sessions"]:
            with self.lock:
                self.error = "nothing was recorded"
            self.hub.publish("state", self.snapshot())

    def _cmd_process(self) -> None:
        sessions = self.scan["sessions"]
        if not sessions:
            raise UsageError("no recorded passes")
        out = Path(self.scan["out"])
        try:
            params = self._build_params(self.scan["name"], self.scan["plan"], str(out))
            params.sessions = [self.sessions_dir / s for s in sessions]
            check_paths(params)
        except UsageError:
            raise
        self.last_out = out
        self.proc.start(sessions=[str(self.sessions_dir / s) for s in sessions], out=out, name=self.scan["name"])
        self._set_state("processing")

    def _cmd_cancel(self) -> None:
        self.proc.cancel()

    def _cmd_reset(self) -> None:
        with self.lock:
            self.scan, self.result, self.error, self.banner = {}, None, "", ""
        self._do_checks()

    def _build_done(self, rc: int, result: dict | None, cancelled: bool, stderr_tail: str) -> None:
        with self.lock:
            if rc == 0 and result and result.get("ok"):
                self.result, self.state = result, "done"
            else:
                msg = (result or {}).get("error") or stderr_tail or f"build exited with code {rc}"
                self.error = "cancelled" if cancelled else msg
                self.state = "scan_done" if cancelled else "failed"
        self.hub.publish("state", self.snapshot())

    # ---------------------------------------------------------------- per-pair work
    def _on_pair(self, pair) -> None:
        now = time.monotonic()
        if self.state == "opening_camera":
            self._begin_pass()
            self._set_state("scanning")
        m, hint = self.coach.update(pair.rgb, pair.depth, now)
        with self.lock:
            self.latest = (pair.rgb, pair.depth)
            self.metrics = {"distance_m": None if m.distance_m is None else round(m.distance_m, 2),
                            "valid_frac": round(m.valid_frac, 2), "sharpness_ratio": round(m.sharpness_ratio, 2),
                            "shift_px": round(m.shift_px, 1), "brightness": round(m.brightness), "hint": hint}
        if self.recorder is not None:
            self.recorder.see(pair)
            self._n_pairs_pass += 1
            if self.scan.get("recording") and (self._n_pairs_pass - 1) % EVERY == 0:
                row = self.recorder.record(pair, self.recorder.t_first or now, now)
                if row is not None:
                    with self.lock:
                        self.scan["frames"] = self.recorder.n_rec
                    if self.recorder.n_rec % 100 == 0 and shutil.disk_usage(self.sessions_dir).free < 1024**3:
                        self._fail("disk almost full: scan stopped", aborted="disk_low")

    def _banner_from_events(self) -> None:
        ev = getattr(self.stream, "stats", None)
        if ev is None:
            return
        for e in ev.events[self._ev_seen:]:
            if e["event"] == "disconnected":
                with self.lock:
                    self.banner = "Kinect disconnected - plug it back in (waiting up to 90 s)"
            elif e["event"] in ("reconnected", "opened"):
                with self.lock:
                    self.banner = ""
        self._ev_seen = len(ev.events)

    def _publish_metrics(self) -> None:
        now = time.monotonic()
        if now - self._last_metrics_t < 0.2 or self.state not in ("scanning", "opening_camera"):
            return
        self._last_metrics_t = now
        with self.lock:
            sc = self.scan
            rec_s = sc.get("rec_s", 0.0) + ((now - self._rec_since) if self._rec_since else 0.0)
            data = dict(self.metrics, frames=sc.get("frames", 0), rec_s=round(rec_s, 1), banner=self.banner,
                        recording=sc.get("recording", False), pass_idx=sc.get("pass_idx", 0))
        self.hub.publish("metrics", data)

    def _fail(self, message: str, aborted: str | None = None) -> None:
        self._close_pass(aborted=aborted)
        if self.stream is not None:
            self.stream.stop()
            self.stream = None
        with self.lock:
            self.error, self.latest = message, None
        self._set_state("scan_done" if self.scan.get("sessions") else "failed")

    def _finalise(self, aborted: str | None = None) -> None:
        try:
            self._close_pass(aborted=aborted)
        finally:
            if self.stream is not None:
                self.stream.stop()
                self.stream = None

    # ---------------------------------------------------------------- video
    def jpeg(self, view: str = "rgb") -> bytes | None:
        fr = self.latest_frame()
        if fr is None:
            return None
        rgb, depth = fr
        if view == "depth":
            from ..preview import depth_colormap
            img = depth_colormap(depth)
        else:
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes() if ok else None
