"""Run `kinect3dgs.dataset build` as a subprocess and turn its JSONL output into events (FR-6, D5)."""
from __future__ import annotations

import json
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path


class ProcessManager:
    def __init__(self, emit, on_exit):
        self.emit, self.on_exit = emit, on_exit
        self.p: subprocess.Popen | None = None
        self.out: Path | None = None
        self.cancelled = False
        self.result: dict | None = None
        self.tail: list[str] = []

    def start(self, sessions: list[str], out: Path, name: str | None = None) -> None:
        self.out, self.cancelled, self.result, self.tail = Path(out), False, None, []
        cmd = [sys.executable, "-m", "kinect3dgs.dataset", "build", "--sessions", *sessions, "--out", str(out),
               "--progress-jsonl", "--quiet"]
        if name:
            cmd += ["--name", name]
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._read_err, daemon=True).start()

    def _read_err(self) -> None:
        for line in self.p.stderr:
            self.tail = (self.tail + [line.rstrip()])[-8:]

    def _read(self) -> None:
        for line in self.p.stdout:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if "result" in obj:
                self.result = obj["result"]
            else:
                self.emit("progress", obj)
        rc = self.p.wait()
        self._cleanup_tmp()
        self.on_exit(rc, self.result, self.cancelled, "\n".join(self.tail))

    def cancel(self) -> None:
        if self.p is None or self.p.poll() is not None:
            return
        self.cancelled = True
        self.p.send_signal(signal.SIGTERM)
        threading.Thread(target=self._escalate, daemon=True).start()

    def _escalate(self) -> None:
        try:
            self.p.wait(5)
        except subprocess.TimeoutExpired:
            self.p.kill()

    def _cleanup_tmp(self) -> None:
        if self.out is None or self.p is None:
            return
        for tmp in self.out.parent.glob(f"{self.out.name}.tmp-{self.p.pid}"):
            shutil.rmtree(tmp, ignore_errors=True)

    def running(self) -> bool:
        return self.p is not None and self.p.poll() is None
