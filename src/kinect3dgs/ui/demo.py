"""Synthetic RGB-D stream with the KinectStream interface, for --demo and tests (FR-12, D9)."""
from __future__ import annotations

import os
import threading
import time

import cv2
import numpy as np

from ..dataset.synthetic import make_scene
from ..stream import FramePair, StreamStats


class DemoStream:
    """Walks a camera back and forth along the synthetic scene at 30 fps, with scripted trouble:
    every `episode_every` seconds, `episode_len` seconds too close, then the same of motion blur."""

    def __init__(self, n_views: int = int(os.environ.get("KINECT3DGS_DEMO_VIEWS", 36)), seed: int = 0, speed: float = 1.0, episode_every: float = 25.0,
                 episode_len: float = 3.0, depth_scale: float = 0.4):
        self.n_views, self.seed, self.speed, self.depth_scale = n_views, seed, speed, depth_scale
        self.episode_every, self.episode_len = episode_every, episode_len
        self.stats = StreamStats()
        self.killed = False
        self.views = None
        self._q: list = []
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._thread = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(2.0)

    def _episode(self, t: float):
        ph = t % self.episode_every
        if self.episode_every - 2 * self.episode_len <= ph < self.episode_every - self.episode_len:
            return "close"
        if self.episode_every - self.episode_len <= ph:
            return "blur"
        return None

    def _run(self) -> None:
        scene = make_scene(self.n_views, 2.0, self.seed)
        # depth scaled uniformly so the demo scene sits at a comfortable table distance (~1 m); geometry stays consistent
        self.views = [(i, (d.astype(np.float32) * self.depth_scale).astype(np.uint16)) for i, d in zip(scene.images, scene.depth_mm)]
        n = len(self.views)
        t_start = time.monotonic()
        self.stats.t_first = None
        frame = 0
        while not self._stop.is_set():
            t = (time.monotonic() - t_start) * self.speed
            pos = t * 2.0                                    # ~2 views per second
            k = int(pos) % (2 * n - 2)
            idx = k if k < n else 2 * n - 2 - k
            rgb, depth = self.views[idx]
            ep = self._episode(t)
            if ep == "close":
                depth = (depth.astype(np.float32) * 0.45).astype(np.uint16)
            elif ep == "blur":
                rgb = cv2.GaussianBlur(rgb, (0, 0), 9)
            jitter = np.float32([[1, 0, np.sin(frame * 0.5) * 1.5], [0, 1, np.cos(frame * 0.4) * 1.5]])   # hand shake
            rgb = cv2.warpAffine(rgb, jitter, (640, 480), borderMode=cv2.BORDER_REPLICATE)
            now = time.monotonic()
            pair = FramePair(frame, rgb.copy(), depth.copy(), time.monotonic_ns(), frame * 5_992_740, frame * 5_992_740,
                             4.0, False, 6.0)
            with self._cv:
                self._q.append(pair)
                self._q = self._q[-4:]
                self._cv.notify_all()
            frame += 1
            time.sleep(max(0.0, 1 / 30 - (time.monotonic() - now)))

    def get_pair(self, timeout: float = 1.0):
        with self._cv:
            if not self._q:
                self._cv.wait(timeout)
            if not self._q:
                return None
            pair = self._q.pop(0)
        st = self.stats
        now = time.monotonic()
        st.t_first = st.t_first or now
        st.t_last = now
        st.pairs += 1
        st.skews_ms.append(pair.skew_ms)
        st.host_skews_ms.append(pair.host_skew_ms)
        if st.first_pair_s is None:
            st.first_pair_s = 0.0
        return pair
