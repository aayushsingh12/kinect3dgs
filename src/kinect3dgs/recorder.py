"""SessionRecorder: one AKS-002 session folder, shared by the record CLI and the guided UI (AKS-004 FR-13)."""
from __future__ import annotations

import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .sessions import SCHEMA_VERSION, write_json_atomic
from .writer import PairWriter

REPO = Path(__file__).resolve().parents[2]


def frame_row(idx, p, t0, t_pair):
    gray = cv2.cvtColor(p.rgb, cv2.COLOR_RGB2GRAY)
    return {"idx": idx, "t_host_s": round(t_pair - t0, 4), "ts_rgb": p.ts_rgb, "ts_depth": p.ts_depth,
            "skew_ms": round(p.skew_ms, 3), "host_skew_ms": round(p.host_skew_ms, 3),
            "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
            "depth_valid_frac": round(float((p.depth > 0).mean()), 4), "bad_pair": bool(p.bad)}


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else None


def libfreenect_tag():
    f = REPO / ".local/.installed-tag"
    return f.read_text().strip() if f.exists() else None


class SessionRecorder:
    def __init__(self, root: Path, args: dict, devices=None, lock_requested: bool = False, png_level: int = 3,
                 stream=None):
        self.root = Path(root)
        self.root.mkdir(parents=True)
        self.writer = PairWriter(self.root, png_level=png_level)
        self.started = datetime.now(timezone.utc)
        self.lock_requested = lock_requested
        self.n_rec = 0
        self.n_seen = 0
        self.skews: list[float] = []
        self.host_skews: list[float] = []
        self.bad = 0
        self.t_first = self.t_last = None
        self.base = self._counters(stream)
        self.meta = {
            "schema_version": SCHEMA_VERSION, "started_utc": self.started.isoformat(), "finished_utc": None,
            "aborted": None, "host": {"kernel": platform.release(), "python": sys.version.split()[0]},
            "libfreenect": libfreenect_tag(), "device": [d.__dict__ for d in (devices or [])],
            "formats": {"rgb": {"mode": "VIDEO_RGB", "size": [640, 480], "dtype": "uint8", "channel_order": "RGB"},
                        "depth": {"mode": "DEPTH_REGISTERED", "size": [640, 480], "dtype": "uint16", "units": "mm",
                                  "invalid": 0, "registered_to": "rgb"}},
            "args": args, "exposure_lock": {"requested": lock_requested, "results": None},
            "stats": {}, "events": [], "intrinsics": None,
        }
        write_json_atomic(self.root / "session.json", self.meta)

    @staticmethod
    def _counters(stream):
        if stream is None:
            return {}
        st = stream.stats
        return {"dropped_consumer": st.dropped_consumer, "unmatched": st.unmatched, "restarts": st.restarts}

    def see(self, pair) -> None:
        """Count a pair for the session statistics (call for every pair while the session is open)."""
        now = time.monotonic()
        self.t_first = self.t_first or now
        self.t_last = now
        self.n_seen += 1
        self.bad += int(pair.bad)
        self.skews.append(pair.skew_ms)
        self.host_skews.append(pair.host_skew_ms)

    def record(self, pair, t0, now):
        """Write this pair to disk (the caller decides which pairs). Returns the frame row, or None if dropped."""
        row = frame_row(self.n_rec, pair, t0, now)
        if self.writer.submit(self.n_rec, pair.rgb, pair.depth, row):
            self.n_rec += 1
            return row
        return None

    def close(self, stream=None, aborted=None, hard: bool = False) -> dict:
        if not hard:
            self.writer.close()
        else:
            self.writer.pool.shutdown(wait=False, cancel_futures=True)
            self.writer.jl.close()
        st = stream.stats if stream is not None else None
        elapsed = (self.t_last - self.t_first) if self.t_first and self.t_last else 0.0
        fps = (self.n_seen - 1) / elapsed if self.n_seen > 1 and elapsed > 0 else 0.0
        m = self.meta
        m["finished_utc"] = datetime.now(timezone.utc).isoformat()
        m["aborted"] = aborted
        if st is not None:
            m["exposure_lock"]["results"] = st.lock_results
            m["exposure_lock"]["applied"] = bool(st.lock_results) and all(
                isinstance(v, int) and v >= 0 for v in st.lock_results.values())
            m["events"] = st.events
        sk = np.abs(self.skews)
        m["stats"] = {
            "pairs_produced": self.n_seen, "pairs_recorded": self.writer.written, "bad_pairs": self.bad,
            "unmatched_frames": (st.unmatched - self.base.get("unmatched", 0)) if st else 0,
            "dropped_consumer": (st.dropped_consumer - self.base.get("dropped_consumer", 0)) if st else 0,
            "dropped_writer": self.writer.dropped, "write_errors": self.writer.errors,
            "restarts": (st.restarts - self.base.get("restarts", 0)) if st else 0,
            "producer_killed": bool(stream.killed) if stream is not None else False,
            "mean_fps": round(fps, 2), "duration_s": round(elapsed, 2),
            "first_pair_s": round(st.first_pair_s, 2) if st and st.first_pair_s else None,
            "skew_ms": {"p50": pct(sk, 50), "p95": pct(sk, 95), "max": float(sk.max()) if len(sk) else None},
            "host_skew_ms": {"p50": pct(self.host_skews, 50), "p95": pct(self.host_skews, 95),
                             "min": float(np.min(self.host_skews)) if self.host_skews else None,
                             "max": float(np.max(self.host_skews)) if self.host_skews else None},
            "bytes_written": self.writer.bytes, "mean_write_ms": round(self.writer.mean_write_ms, 2),
        }
        write_json_atomic(self.root / "session.json", m)
        return m
