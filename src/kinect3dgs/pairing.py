"""Greedy one-to-one pairing of RGB and depth frames by *device* timestamp (pure Python).

Both Kinect streams carry timestamps from one device clock; host arrival time only reflects USB delivery
phase (it varies by tens of ms between runs), so it is kept as a diagnostic, not used for matching.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import median
from typing import Any

TS_MASK = 0xFFFFFFFF


def ticks_diff(a: int, b: int) -> int:
    """Signed b - a on the 32-bit device counter."""
    d = (b - a) & TS_MASK
    return d - (TS_MASK + 1) if d > TS_MASK // 2 else d


@dataclass
class Frame:
    t_ns: int          # host monotonic arrival time (diagnostic only)
    ts_dev: int        # device timestamp
    data: Any          # ndarray


@dataclass
class Pair:
    rgb: Frame
    depth: Frame
    skew_ms: float       # depth - rgb on the device clock, in ms (assumes the nominal frame rate)
    host_skew_ms: float  # depth - rgb host arrival difference (diagnostic)
    bad: bool            # |skew_ms| above the reporting threshold


class Pairer:
    def __init__(self, nominal_fps: float = 30.0, accept_frac: float = 0.5, bad_ms: float = 15.0, keep: int = 4):
        self.nominal_ms = 1000.0 / nominal_fps
        self.accept_frac, self.bad_ms, self.keep = accept_frac, bad_ms, keep
        self.rgb: deque[Frame] = deque()
        self.depth: deque[Frame] = deque()
        self.unmatched = 0
        self.period: float | None = None          # device ticks per frame, estimated from the RGB stream
        self._prev_rgb_ts: int | None = None
        self._diffs: deque[int] = deque(maxlen=15)

    def push_rgb(self, f: Frame) -> list[Pair]:
        if self._prev_rgb_ts is not None:
            d = ticks_diff(self._prev_rgb_ts, f.ts_dev)
            if d > 0:
                self._diffs.append(d)
                self.period = float(median(self._diffs))
        self._prev_rgb_ts = f.ts_dev
        self._push(self.rgb, f)
        return self._match()

    def push_depth(self, f: Frame) -> list[Pair]:
        self._push(self.depth, f)
        return self._match()

    def _push(self, dq: deque, f: Frame) -> None:
        dq.append(f)
        while len(dq) > self.keep:
            dq.popleft()
            self.unmatched += 1

    def _match(self) -> list[Pair]:
        out: list[Pair] = []
        if self.period is None:          # need two RGB frames to know the frame period
            return out
        accept = self.accept_frac * self.period
        while self.rgb and self.depth:
            dt = ticks_diff(self.rgb[0].ts_dev, self.depth[0].ts_dev)
            if abs(dt) <= accept:
                r, d = self.rgb.popleft(), self.depth.popleft()
                skew = dt / self.period * self.nominal_ms
                out.append(Pair(r, d, skew, (d.t_ns - r.t_ns) / 1e6, abs(skew) > self.bad_ms))
            elif dt > 0:
                self.rgb.popleft()       # rgb head is older: nothing can match it any more
                self.unmatched += 1
            else:
                self.depth.popleft()
                self.unmatched += 1
        return out
