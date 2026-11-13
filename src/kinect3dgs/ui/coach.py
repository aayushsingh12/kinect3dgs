"""Coaching rules (FR-5): pure functions over the latest frame and a short history. One hint at a time."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import median

import cv2
import numpy as np

from .plans import Pass

CENTRE = (slice(144, 336), slice(192, 448))
HOLD_S = 0.5
# hint id -> (priority, text); lower priority number wins
HINTS = {
    "no_depth": (1, "no depth: too close, too far or facing glass"),
    "too_close": (2, "too close"),
    "too_fast": (3, "slow down"),
    "blurry": (4, "blurry - slow down"),
    "too_far": (5, "too far"),
    "dark": (6, "too dark"),
    "bright": (7, "too bright"),
}


@dataclass
class Metrics:
    distance_m: float | None
    valid_frac: float
    sharpness_ratio: float
    shift_px: float
    brightness: float
    saturated_frac: float


class Coach:
    def __init__(self, plan_pass: Pass, fps: int = 30):
        self.pp = plan_pass
        self.hist: deque[float] = deque(maxlen=4 * fps)
        self.prev = None
        self.shift = 0.0
        self.since: dict[str, float] = {}      # hint -> time the condition started holding
        self.clear_since: dict[str, float] = {}
        self.shown: str | None = None

    def set_pass(self, plan_pass: Pass) -> None:
        self.pp = plan_pass
        self.since.clear(); self.clear_since.clear(); self.shown = None

    def measure(self, rgb: np.ndarray, depth: np.ndarray) -> Metrics:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        ratio = sharp / median(self.hist) if self.hist and median(self.hist) > 0 else 1.0
        self.hist.append(sharp)
        small = cv2.resize(gray, (160, 120)).astype(np.float32)
        if self.prev is not None and small.std() > 2.0 and self.prev.std() > 2.0:      # needs texture to track motion
            (dx, dy), _ = cv2.phaseCorrelate(self.prev, small)
            step = float(np.hypot(dx, dy) * 4.0)
            self.shift = 0.3 * (step if np.isfinite(step) else 0.0) + 0.7 * self.shift
        elif self.prev is not None:
            self.shift *= 0.7
        self.prev = small
        centre = depth[CENTRE]
        valid = centre[centre > 0]
        dist = float(np.median(valid)) / 1000.0 if valid.size > 200 else None
        return Metrics(dist, float((depth > 0).mean()), ratio, self.shift, float(gray.mean()),
                       float((gray >= 250).mean()))

    def conditions(self, m: Metrics) -> set[str]:
        z = self.pp.zone
        active = set()
        if m.valid_frac < 0.15 or m.distance_m is None:
            active.add("no_depth")
        elif m.distance_m < z.warn[0] or m.distance_m < z.ideal[0] - 0.05:
            active.add("too_close")
        elif m.distance_m > z.warn[1] or m.distance_m > z.ideal[1] + 0.05:
            active.add("too_far")
        if m.shift_px > self.pp.speed_limit_px:
            active.add("too_fast")
        elif m.sharpness_ratio < 0.5:
            active.add("blurry")
        if m.brightness < 45:
            active.add("dark")
        elif m.brightness > 215 or m.saturated_frac > 0.15:
            active.add("bright")
        return active

    def hint(self, active: set[str], t: float) -> str | None:
        """Hysteresis: a condition must hold HOLD_S to appear and be absent HOLD_S to clear; one text at a time."""
        for h in HINTS:
            if h in active:
                self.since.setdefault(h, t)
                self.clear_since.pop(h, None)
            else:
                self.since.pop(h, None)
                self.clear_since.setdefault(h, t)
        confirmed = [h for h in HINTS if h in active and t - self.since[h] >= HOLD_S]
        if self.shown and self.shown not in active and t - self.clear_since.get(self.shown, t) < HOLD_S:
            return HINTS[self.shown][1]          # keep it for a moment after the condition clears
        self.shown = min(confirmed, key=lambda h: HINTS[h][0]) if confirmed else None
        return HINTS[self.shown][1] if self.shown else None

    def update(self, rgb, depth, t: float):
        m = self.measure(rgb, depth)
        return m, self.hint(self.conditions(m), t)
