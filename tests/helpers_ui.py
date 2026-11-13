"""Helpers for UI tests: a fast fake stream with the KinectStream interface."""
import threading
import time

import numpy as np

from kinect3dgs.errors import KinectTimeoutError
from kinect3dgs.stream import FramePair, StreamStats


class FakeStream:
    def __init__(self, views=None, fps=60, depth_mm=1100, brightness=120, events=None, fail_after=None):
        rng = np.random.default_rng(0)
        base = rng.integers(40, 200, (480, 640, 3), dtype=np.uint8)
        self.views = views or [(np.roll(base, i * 3, axis=1).copy(), np.full((480, 640), depth_mm, np.uint16)) for i in range(8)]
        self.fps, self.killed = fps, False
        self.stats = StreamStats()
        self.stats.events = list(events or [])
        self.fail_after = fail_after
        self.n = 0
        self.t_last = time.monotonic()
        self.started = self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def get_pair(self, timeout=1.0):
        if self.fail_after is not None and self.n >= self.fail_after:
            raise KinectTimeoutError("fake stall")
        dt = 1 / self.fps - (time.monotonic() - self.t_last)
        if dt > 0:
            time.sleep(min(dt, timeout))
        self.t_last = time.monotonic()
        rgb, depth = self.views[self.n % len(self.views)]
        p = FramePair(self.n, rgb.copy(), depth.copy(), time.monotonic_ns(), self.n * 6_000_000, self.n * 6_000_000, 3.0, False, 5.0)
        st = self.stats
        st.t_first = st.t_first or self.t_last
        st.t_last = self.t_last
        st.pairs += 1
        st.skews_ms.append(3.0)
        st.host_skews_ms.append(5.0)
        self.n += 1
        return p
