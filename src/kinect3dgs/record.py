"""Record an RGB-D session: python -m kinect3dgs.record [--seconds S] [--every K] ..."""
from __future__ import annotations

import argparse
import platform
import select
import shutil
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from . import sensor
from .errors import KinectError, KinectNotFoundError, KinectTimeoutError
from .recorder import SessionRecorder, frame_row, pct  # noqa: F401  (re-exported for compatibility)
from .stream import KinectStream

MB_PER_PAIR = 0.5          # measured on hardware (PNG level 3, indoor scene); used for the disk pre-check only
MIN_FREE_BYTES = 2 * 1024**3


def parse(argv=None):
    ap = argparse.ArgumentParser(prog="kinect3dgs.record", description=__doc__)
    ap.add_argument("--mode", choices=["continuous", "snapshot"], default="continuous")
    ap.add_argument("--seconds", type=float, default=None, help="stop after S seconds of recording")
    ap.add_argument("--frames", type=int, default=None, help="stop after N recorded pairs")
    ap.add_argument("--every", type=int, default=3, help="record every K-th pair (30 fps / K)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", type=Path, default=Path("data/sessions"))
    ap.add_argument("--exposure", choices=["lock", "auto"], default="auto",
                    help="auto (default). lock = ask the camera to switch off auto exposure/white balance/flicker after "
                         "--lock-after seconds; EXPERIMENTAL: the calls succeed on the 1473 but exposure still adapts (AKS-002 R7)")
    ap.add_argument("--lock-after", type=float, default=3.0)
    ap.add_argument("--warmup", type=float, default=None, help="seconds before recording starts (default: lock-after+1 or 1)")
    ap.add_argument("--png-level", type=int, default=3)
    ap.add_argument("--preview", nargs="?", const=8765, type=int, default=None, metavar="PORT")
    return ap.parse_args(argv)


def free_bytes(path: Path) -> int:
    p = path
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


def estimate_bytes(args) -> int:
    if args.mode == "snapshot":
        return int(60 * MB_PER_PAIR * 1e6)
    if args.frames:
        pairs = args.frames
    else:
        pairs = (args.seconds or 600.0) * 30.0 / max(1, args.every)
    return int(pairs * MB_PER_PAIR * 1e6)


def main(argv=None) -> int:
    args = parse(argv)
    if args.every < 1:
        print("--every must be >= 1")
        return 2
    name = args.name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = args.out / name
    if root.exists():
        print(f"refusing to overwrite existing session {root}")
        return 2
    need = max(MIN_FREE_BYTES, 3 * estimate_bytes(args))
    free = free_bytes(args.out)
    print(f"estimated size {estimate_bytes(args) / 1e9:.2f} GB, free {free / 1e9:.1f} GB")
    if free < need:
        print(f"not enough free disk space (need {need / 1e9:.1f} GB)")
        return 2

    devs = sensor.list_devices()
    if not any(d.pid in sensor.CAMERA_PIDS for d in devs):
        print("No Kinect camera on USB.", sensor.L0_HINT)
        return 2

    stop, hard = threading.Event(), threading.Event()
    last_sig = [0.0]

    def on_sig(signum, frame):
        now = time.monotonic()
        if stop.is_set() and now - last_sig[0] < 2.0:
            hard.set()
        stop.set()
        last_sig[0] = now

    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)

    lock = args.exposure == "lock"
    warmup = args.warmup if args.warmup is not None else (args.lock_after + 1.0 if lock else 1.0)
    rec = SessionRecorder(root, {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                          devices=devs, lock_requested=lock, png_level=args.png_level)

    preview = None
    if args.preview is not None:
        from .preview import Preview
        preview = Preview(args.preview)
        print("preview:", preview.start())

    want_snap, quit_evt = threading.Event(), threading.Event()
    if args.mode == "snapshot":
        def stdin_loop():
            while not stop.is_set() and not quit_evt.is_set():
                try:
                    r, _, _ = select.select([sys.stdin], [], [], 0.2)
                except (ValueError, OSError):
                    return
                if r:
                    line = sys.stdin.readline()
                    if line == "" or line.strip().lower() == "q":
                        quit_evt.set()
                    else:
                        want_snap.set()
        threading.Thread(target=stdin_loop, daemon=True).start()
        print("Snapshot mode: press Enter to save a pair, q + Enter to finish.")

    stream = KinectStream(lock_exposure=lock, lock_after_s=args.lock_after)
    rec.base = rec._counters(stream)
    rc, aborted = 0, None
    n_pairs = 0
    t0 = time.monotonic()
    try:
        stream.start()
        t_rec0 = None
        while not stop.is_set() and not quit_evt.is_set() and not hard.is_set():
            p = stream.get_pair(0.5)
            if p is None:
                continue
            now = time.monotonic()
            n_pairs += 1
            rec.see(p)
            if preview:
                preview.update(p.rgb, p.depth)
            if t_rec0 is None and now - t0 >= warmup:
                t_rec0 = now
            if t_rec0 is None and args.mode == "continuous":
                continue
            take = False
            if args.mode == "continuous":
                take = (n_pairs - 1) % args.every == 0
            elif want_snap.is_set():
                want_snap.clear()
                take = True
            if take:
                row = rec.record(p, t_rec0 or t0, now)
                if row is not None:
                    if args.mode == "snapshot":
                        print(f"saved {row['idx']:06d}  depth valid {row['depth_valid_frac']:.0%}  sharpness {row['sharpness']:.0f}")
                    elif rec.n_rec % 30 == 0:
                        print(f"{rec.n_rec} pairs recorded, {stream.stats.mean_fps:.1f} fps")
                if rec.n_rec % 100 == 0 and rec.n_rec and free_bytes(args.out) < 1024**3:
                    aborted, rc = "disk_low", 1
                    break
            if t_rec0 is not None and args.seconds and now - t_rec0 >= args.seconds:
                break
            if args.frames and rec.n_rec >= args.frames:
                break
        if stop.is_set() or hard.is_set():
            aborted, rc = aborted or "signal", 1 if aborted else 0
    except KinectNotFoundError as e:
        print("FAIL:", e, "\n", getattr(e, "hint", ""))
        aborted, rc = "not_found", 2
    except KinectTimeoutError as e:
        print("FAIL:", e)
        aborted, rc = "stalled", 3
    except KinectError as e:
        print("FAIL:", e)
        aborted, rc = "error", 3
    finally:
        stream.stop()
        if preview:
            preview.stop()

    meta = rec.close(stream, aborted, hard=hard.is_set())
    s = meta["stats"]
    print(f"\nsession {root}: {s['pairs_recorded']} pairs recorded of {s['pairs_produced']} produced, "
          f"{s['mean_fps']} fps, skew p95 {s['skew_ms']['p95']} ms, dropped consumer/writer "
          f"{s['dropped_consumer']}/{s['dropped_writer']}, {s['bytes_written'] / 1e6:.0f} MB")
    if aborted and aborted != "signal":
        print("ended early:", aborted)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
