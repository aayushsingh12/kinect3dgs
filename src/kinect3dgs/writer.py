"""Threaded PNG writer with back-pressure (drops and counts instead of stalling the stream)."""
from __future__ import annotations

import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from .sessions import depth_path, rgb_path


class PairWriter:
    def __init__(self, root: Path, threads: int = 3, png_level: int = 3, queue_max: int = 32, flush_every: int = 50):
        self.root = Path(root)
        (self.root / "rgb").mkdir(parents=True, exist_ok=True)
        (self.root / "depth").mkdir(parents=True, exist_ok=True)
        self.level = png_level
        self.queue_max = queue_max
        self.flush_every = flush_every
        self.pool = ThreadPoolExecutor(max_workers=threads)
        self.inflight: deque = deque()
        self.jl = open(self.root / "frames.jsonl", "a", buffering=1)
        self.dropped = 0
        self.errors = 0
        self.written = 0
        self.bytes = 0
        self.write_ms_sum = 0.0

    def _encode(self, idx, rgb, depth):
        t0 = time.perf_counter()
        p = [cv2.IMWRITE_PNG_COMPRESSION, self.level]
        a, b = rgb_path(self.root, idx), depth_path(self.root, idx)
        cv2.imwrite(str(a), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), p)
        cv2.imwrite(str(b), depth, p)
        return (time.perf_counter() - t0) * 1000.0, a.stat().st_size + b.stat().st_size

    def submit(self, idx: int, rgb, depth, row: dict) -> bool:
        self.reap()
        if len(self.inflight) >= self.queue_max:
            self.dropped += 1
            return False
        self.inflight.append((row, self.pool.submit(self._encode, idx, rgb, depth)))
        return True

    def reap(self, wait: bool = False) -> None:
        while self.inflight and (wait or self.inflight[0][1].done()):
            row, fut = self.inflight.popleft()
            try:
                ms, nbytes = fut.result()
            except Exception:
                self.errors += 1
                continue
            self.jl.write(json.dumps(row) + "\n")
            self.written += 1
            self.bytes += nbytes
            self.write_ms_sum += ms

    def close(self) -> None:
        self.reap(wait=True)
        self.pool.shutdown(wait=True)
        self.jl.close()

    @property
    def mean_write_ms(self) -> float:
        return self.write_ms_sum / self.written if self.written else 0.0
