"""Long-lived RGB + registered-depth streaming from the Kinect (async libfreenect API in a child process)."""
from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import secrets
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import sensor
from .errors import KinectNotFoundError, KinectTimeoutError
from .pairing import Frame, Pairer
from .ring import PREFIX, Ring

FLAG_NAMES = ("AUTO_EXPOSURE", "AUTO_WHITE_BALANCE", "AUTO_FLICKER")


@dataclass
class FramePair:
    seq: int
    rgb: np.ndarray
    depth: np.ndarray
    t_host_ns: int
    ts_rgb: int
    ts_depth: int
    skew_ms: float
    bad: bool
    host_skew_ms: float = 0.0


@dataclass
class StreamStats:
    pairs: int = 0
    bad_pairs: int = 0
    dropped_consumer: int = 0
    unmatched: int = 0
    restarts: int = 0
    first_pair_s: float | None = None
    t_first: float | None = None
    t_last: float | None = None
    skews_ms: list = field(default_factory=list)
    host_skews_ms: list = field(default_factory=list)
    events: list = field(default_factory=list)
    lock_results: dict | None = None
    registered_depth: bool = True

    @property
    def mean_fps(self) -> float:
        if self.pairs < 2 or self.t_last is None or self.t_first is None or self.t_last <= self.t_first:
            return 0.0
        return (self.pairs - 1) / (self.t_last - self.t_first)


def _producer(n_slots: int, ring_name: str, conn, stop_evt, lock_exposure: bool, lock_after_s: float,
              registered: bool) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)   # the parent owns shutdown
    ring = None
    ctx = dev = None
    try:
        import freenect
        ctx = freenect.init()
        dev = freenect.open_device(ctx, 0) if ctx is not None else None
        if dev is None:
            conn.send(("error", "open"))
            return
        depth_mode = freenect.DEPTH_REGISTERED if registered else freenect.DEPTH_MM
        freenect.set_depth_mode(dev, freenect.RESOLUTION_MEDIUM, depth_mode)
        freenect.set_video_mode(dev, freenect.RESOLUTION_MEDIUM, freenect.VIDEO_RGB)
        ring = Ring(n_slots, ring_name)
        pairer = Pairer()
        state = {"seq": 0, "t_start": None, "locked": False}

        def emit(pairs):
            for p in pairs:
                slot = ring.publish(state["seq"], p.rgb.data, p.depth.data, p.depth.t_ns, p.rgb.ts_dev,
                                    p.depth.ts_dev, p.skew_ms, int(p.bad), p.host_skew_ms)
                conn.send(("pair", slot, state["seq"], p.skew_ms, p.bad, pairer.unmatched))
                state["seq"] += 1

        def on_video(d, arr, ts):
            t = time.monotonic_ns()
            if state["t_start"] is None:
                state["t_start"] = time.monotonic()
            emit(pairer.push_rgb(Frame(t, int(ts), np.array(arr, copy=True))))

        def on_depth(d, arr, ts):
            t = time.monotonic_ns()
            emit(pairer.push_depth(Frame(t, int(ts), np.array(arr, copy=True))))

        def body(d, c):
            if lock_exposure and not state["locked"] and state["t_start"] is not None \
                    and time.monotonic() - state["t_start"] >= lock_after_s:
                res = {}
                for name in FLAG_NAMES:
                    try:   # libfreenect returns the reply length (>= 0) on success, a negative error code otherwise
                        res[name] = int(freenect.set_flag(d, getattr(freenect, "FLAG_" + name), freenect.FLAG_OFF))
                    except Exception as e:       # old wrapper without set_flag
                        res[name] = f"unsupported: {e}"
                conn.send(("lock", res))
                state["locked"] = True
            if stop_evt.is_set():
                raise freenect.Kill()

        conn.send(("opened",))
        freenect.runloop(depth=on_depth, video=on_video, body=body, dev=dev)
    except Exception as e:  # pragma: no cover - reported to parent
        try:
            conn.send(("error", repr(e)))
        except Exception:
            pass
    finally:
        try:
            if dev is not None:
                freenect.close_device(dev)
            if ctx is not None:
                freenect.shutdown(ctx)
        except Exception:
            pass
        if ring is not None:
            ring.close()
        try:
            conn.send(("exited",))
        except Exception:
            pass


class KinectStream:
    def __init__(self, n_slots: int = 8, lock_exposure: bool = False, lock_after_s: float = 3.0,
                 open_timeout: float = 45.0, stall_timeout: float = 5.0, max_restarts: int = 2,
                 registered: bool = True, usb_wait: float = 90.0, _target=_producer, _reset=sensor.reset_camera):
        self.n_slots, self.lock_exposure, self.lock_after_s = n_slots, lock_exposure, lock_after_s
        self.open_timeout, self.stall_timeout, self.max_restarts = open_timeout, stall_timeout, max_restarts
        self.registered = registered
        self.usb_wait = usb_wait
        self._wait_usb_deadline = None
        self._target, self._reset = _target, _reset
        self.stats = StreamStats(registered_depth=registered)
        self.ring: Ring | None = None
        self._proc = None
        self._conn = None
        self._stop_evt = None
        self._pending: deque = deque()
        self._started = False
        self._stopped = False
        self._t_spawn = 0.0
        self._t_start = 0.0
        self.killed = False

    # --- lifecycle -------------------------------------------------------------
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *a):
        self.stop()

    def start(self) -> None:
        if self._started:
            return
        self.ring = Ring(self.n_slots, name=f"{PREFIX}{os.getpid()}-{secrets.token_hex(3)}", create=True)
        atexit.register(self._cleanup)
        self._started = True
        self._t_start = time.monotonic()
        self._spawn()

    def _spawn(self) -> None:
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe(duplex=False)
        self._stop_evt = ctx.Event()
        self.ring.reset()
        self._pending.clear()
        self._conn = parent
        self._proc = ctx.Process(
            target=self._target, daemon=True,
            args=(self.n_slots, self.ring.name, child, self._stop_evt, self.lock_exposure,
                  self.lock_after_s, self.registered))
        self._proc.start()
        child.close()
        self._t_spawn = time.monotonic()
        self._last_activity = self._t_spawn

    def _stop_producer(self, grace: float = 5.0) -> None:
        if self._proc is None:
            return
        self._stop_evt.set()
        self._proc.join(grace)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(2.0)
        if self._proc.is_alive():
            self._proc.kill()
            self._proc.join()
            self.killed = True
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
            self._conn = None
        self._proc = None

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._stop_producer()
        self._cleanup()

    def _cleanup(self) -> None:
        if self.ring is not None:
            ring, self.ring = self.ring, None
            ring.close()
            ring.unlink()

    # --- consuming ---------------------------------------------------------------
    def _drain(self) -> None:
        while self._conn is not None and self._conn.poll(0):
            try:
                msg = self._conn.recv()
            except (EOFError, OSError):
                return
            kind = msg[0]
            if kind == "pair":
                self._pending.append(msg)
                # slots older than n-2 may already be overwritten: drop them without reading
                while len(self._pending) > self.n_slots - 2:
                    self._pending.popleft()
                    self.stats.dropped_consumer += 1
            elif kind == "lock":
                self.stats.lock_results = msg[1]
            elif kind == "error":
                self._error = msg[1]
            elif kind == "opened":
                self.stats.events.append({"t": round(time.monotonic() - self._t_start, 2), "event": "opened"})

    def get_pair(self, timeout: float = 1.0) -> FramePair | None:
        deadline = time.monotonic() + timeout
        while True:
            self._drain()
            while self._pending:
                _, slot, seq, skew, bad, unmatched = self._pending.popleft()
                r = self.ring.read(slot, seq)
                if r is None:
                    self.stats.dropped_consumer += 1
                    continue
                rgb, depth, h = r
                now = time.monotonic()
                self._last_activity = now
                st = self.stats
                if st.t_first is None:
                    st.t_first = now
                    st.first_pair_s = now - self._t_start
                st.t_last = now
                st.pairs += 1
                st.bad_pairs += int(bad)
                st.unmatched = unmatched
                st.skews_ms.append(skew)
                st.host_skews_ms.append(h[6] / 1000.0)
                return FramePair(seq, rgb, depth, int(h[1]), int(h[2]), int(h[3]), float(skew), bool(bad), h[6] / 1000.0)
            self._check_watchdog()
            if time.monotonic() >= deadline:
                return None
            if self._conn is not None:
                self._conn.poll(min(0.05, max(0.0, deadline - time.monotonic())))
            else:
                time.sleep(0.05)

    # --- watchdog ----------------------------------------------------------------
    @staticmethod
    def _camera_on_usb() -> bool:
        return any(d.pid in sensor.CAMERA_PIDS for d in sensor.list_devices())

    def _event(self, name: str, **kw) -> None:
        self.stats.events.append({"t": round(time.monotonic() - self._t_start, 2), "event": name, **kw})

    def _check_watchdog(self) -> None:
        now = time.monotonic()
        if self._wait_usb_deadline is not None:            # waiting for the Kinect to be plugged back in
            if self._camera_on_usb():
                time.sleep(2.0)                              # let the device settle after enumeration
                self._wait_usb_deadline = None
                self._event("reconnected")
                print("Kinect is back, resuming.", file=sys.stderr, flush=True)
                self._spawn()
            elif now > self._wait_usb_deadline:
                self.stop()
                raise KinectNotFoundError(f"Kinect did not come back within {self.usb_wait:.0f}s", sensor.L0_HINT)
            return
        dead = self._proc is not None and not self._proc.is_alive() and not self._pending
        limit = self.open_timeout if self.stats.pairs == 0 else self.stall_timeout
        stalled = now - self._last_activity > limit
        if dead or stalled:
            reason = "producer died" if dead else f"no pair for {now - self._last_activity:.0f}s"
            self._restart_or_raise(reason)

    def _restart_or_raise(self, reason: str) -> None:
        if self.stats.pairs > 0 and not self._camera_on_usb():
            # the device vanished from USB (cable, power): wait for it instead of burning restarts
            self._stop_producer(grace=2.0)
            self._wait_usb_deadline = time.monotonic() + self.usb_wait
            self._event("disconnected", reason=reason)
            print(f"Kinect disconnected ({reason}). Check the cable and AC adapter and plug it back in; "
                  f"waiting up to {self.usb_wait:.0f}s ...", file=sys.stderr, flush=True)
            return
        self._event("restart", reason=reason)
        if self.stats.restarts >= self.max_restarts:
            self.stop()
            if self.stats.pairs == 0:
                raise KinectNotFoundError(f"could not start the stream ({reason})", sensor.L0_HINT)
            raise KinectTimeoutError(f"stream stalled and restarts exhausted ({reason})")
        self.stats.restarts += 1
        self._stop_producer(grace=3.0)
        try:
            self._reset()
        except Exception:
            pass
        self._spawn()
