"""Progress events (JSON lines or callback) and cancellation (FR-11)."""
from __future__ import annotations

import json
import signal
import sys
import threading
import time
from contextlib import contextmanager

from .errors import Cancelled

STAGES = ("select", "features", "matching", "mapping", "scale", "undistort", "export", "validate")


class Emitter:
    def __init__(self, callback=None, jsonl: bool = False, quiet: bool = False, stream=None):
        self.callback, self.jsonl, self.quiet = callback, jsonl, quiet
        self.stream = stream or sys.stdout
        self.t0 = time.monotonic()
        self.cancel = threading.Event()
        self.events: list[dict] = []

    def event(self, stage: str, status: str, fraction=None, message: str = "") -> None:
        ev = {"stage": stage, "status": status, "fraction": fraction, "message": message,
              "elapsed_s": round(time.monotonic() - self.t0, 2)}
        self.events.append(ev)
        if self.callback:
            self.callback(ev)
        if self.jsonl:
            print(json.dumps(ev), file=self.stream, flush=True)
        elif not self.quiet and status in ("start", "done", "error"):
            print(f"[{ev['elapsed_s']:7.1f}s] {stage:9s} {status:5s} {message}", file=self.stream, flush=True)

    def result(self, payload: dict) -> None:
        if self.jsonl:
            print(json.dumps({"result": payload}), file=self.stream, flush=True)

    def check(self) -> None:
        if self.cancel.is_set():
            raise Cancelled("cancelled")

    def install_signals(self) -> None:
        def handler(signum, frame):
            self.cancel.set()
        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

    @contextmanager
    def stage(self, name: str, message: str = ""):
        self.check()
        self.event(name, "start", None, message)
        try:
            yield
        except Cancelled:
            self.event(name, "error", None, "cancelled")
            raise
        except Exception as e:
            self.event(name, "error", None, str(e))
            raise
        self.event(name, "done", 1.0)
