"""Hardware smoke test: python -m kinect3dgs.probe [--frames N] [--out DIR] [--timeout S]."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from . import checks, sensor
from .errors import KinectBusyError, KinectNotFoundError, KinectTimeoutError


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="kinect3dgs.probe")
    ap.add_argument("--frames", type=int, default=15, help="warm-up frames to discard")
    ap.add_argument("--out", type=Path, default=Path("data/probe"))
    ap.add_argument("--timeout", type=float, default=30.0, help="seconds; first open of a 1473 unit takes ~6 s")
    args = ap.parse_args(argv)

    devs = sensor.list_devices()
    print("Microsoft USB devices:")
    for d in devs:
        print(f"  {d.bus_id}  {d.vid}:{d.pid}  {d.product}")
    if not any(d.pid in sensor.CAMERA_PIDS for d in devs):
        print("L0 FAIL:", sensor.L0_HINT)
        return 2
    try:
        cap = sensor.capture(args.frames, 3.0, args.timeout)
    except KinectNotFoundError as e:
        print("FAIL:", e, "\n", e.hint)
        return 2
    except (KinectBusyError, KinectTimeoutError) as e:
        print("FAIL:", e)
        return 3

    out = args.out / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "rgb.png"), cv2.cvtColor(cap.rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "depth_mm.png"), cap.depth_mm)
    np.save(out / "depth_mm.npy", cap.depth_mm)
    valid = cap.depth_mm > 0
    vis = np.zeros((*cap.depth_mm.shape, 3), np.uint8)
    if valid.any():
        d = cap.depth_mm[valid].astype(np.float32)
        norm = np.clip((d - d.min()) / max(float(d.max() - d.min()), 1.0) * 255, 0, 255).astype(np.uint8)
        col = cv2.applyColorMap(norm.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
        vis[valid] = col
    cv2.imwrite(str(out / "depth_vis.png"), vis)

    results = [checks.check_shapes(cap.rgb, cap.depth_mm), checks.check_rgb_content(cap.rgb),
               checks.check_depth_content(cap.depth_mm), checks.check_stream(cap.stamps, cap.seconds),
               checks.check_roundtrip(out, cap.rgb, cap.depth_mm)]
    print(f"\nOutput: {out}")
    for r in results:
        print(f"  {r.rung}  {'PASS' if r.passed else 'FAIL'}  {r.name}: {r.detail}")
    meta = {"utc": datetime.now(timezone.utc).isoformat(), "kernel": platform.release(),
            "python": sys.version.split()[0], "numpy": np.__version__, "cv2": cv2.__version__,
            "devices": [d.__dict__ for d in devs],
            "checks": [r.__dict__ for r in results]}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    ok = all(r.passed for r in results)
    print("\nRESULT:", "ALL PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
