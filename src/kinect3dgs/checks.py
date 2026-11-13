"""Pure sanity checks on captured data (no hardware needed)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

H, W = 480, 640
RGB_STD_MIN = 5.0
RGB_MEAN_RANGE = (5.0, 250.0)
DEPTH_VALID_FRAC_MIN = 0.10
DEPTH_RANGE_MM = (400, 6000)   # sanity bound; accuracy is only good below ~4 m
DEPTH_GOOD_ACCURACY_MM = 4000
DEPTH_IN_RANGE_FRAC_MIN = 0.90
FPS_MIN = 20.0
TS_WRAP = 2**32


@dataclass
class CheckResult:
    rung: str
    name: str
    passed: bool
    detail: str


def check_shapes(rgb: np.ndarray, depth: np.ndarray) -> CheckResult:
    ok = (rgb.shape == (H, W, 3) and rgb.dtype == np.uint8
          and depth.shape == (H, W) and depth.dtype == np.uint16)
    return CheckResult("L3", "frame shapes", ok,
                       f"rgb {rgb.shape} {rgb.dtype}, depth {depth.shape} {depth.dtype}")


def check_rgb_content(rgb: np.ndarray) -> CheckResult:
    std = rgb.reshape(-1, rgb.shape[-1]).std(axis=0)
    mean = float(rgb.mean())
    ok = bool((std > RGB_STD_MIN).all() and RGB_MEAN_RANGE[0] < mean < RGB_MEAN_RANGE[1])
    return CheckResult("L4", "rgb content", ok,
                       f"std per channel {np.round(std, 1).tolist()}, mean {mean:.1f}")


def check_depth_content(depth: np.ndarray) -> CheckResult:
    valid = depth > 0
    vfrac = float(valid.mean())
    if not valid.any():
        return CheckResult("L5", "depth content", False,
                           "all depth pixels invalid (0): too close, sunlight, or IR blocked")
    lo, hi = DEPTH_RANGE_MM
    inr = float(((depth[valid] >= lo) & (depth[valid] <= hi)).mean())
    ok = vfrac >= DEPTH_VALID_FRAC_MIN and inr >= DEPTH_IN_RANGE_FRAC_MIN
    detail = (f"valid {vfrac:.1%} (need >= {DEPTH_VALID_FRAC_MIN:.0%}), "
              f"in {lo}-{hi} mm {inr:.1%} (need >= {DEPTH_IN_RANGE_FRAC_MIN:.0%}), "
              f"min/median/max {int(depth[valid].min())}/{int(np.median(depth[valid]))}/{int(depth[valid].max())} mm")
    far = float((depth[valid] > DEPTH_GOOD_ACCURACY_MM).mean())
    if far > 0.25:
        detail += f"; note: {far:.0%} of valid pixels are beyond 4 m, where Kinect accuracy degrades"
    if not ok:
        detail += "; point the sensor at a room 1-3 m away"
    return CheckResult("L5", "depth content", ok, detail)


def check_stream(stamps: list[int], seconds: float) -> CheckResult:
    n = len(stamps)
    fps = n / seconds if seconds > 0 else 0.0
    diffs = [(b - a) % TS_WRAP for a, b in zip(stamps, stamps[1:])]
    mono = n >= 2 and all(0 < d < TS_WRAP // 2 for d in diffs)
    return CheckResult("L6", "streaming", fps >= FPS_MIN and mono,
                       f"{fps:.1f} fps over {seconds:.1f}s ({n} frames), timestamps monotonic: {mono}")


def check_roundtrip(out_dir: Path, rgb: np.ndarray, depth: np.ndarray) -> CheckResult:
    png_rgb = cv2.cvtColor(cv2.imread(str(out_dir / "rgb.png"), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    png_d = cv2.imread(str(out_dir / "depth_mm.png"), cv2.IMREAD_UNCHANGED)
    npy_d = np.load(out_dir / "depth_mm.npy")
    ok = bool(np.array_equal(png_rgb, rgb) and np.array_equal(png_d, depth) and np.array_equal(npy_d, depth))
    return CheckResult("L7", "file round-trip", ok, "png/npy equal in-memory arrays" if ok else "mismatch")
