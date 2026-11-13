"""Single-producer / single-consumer shared-memory ring of RGB+depth slots with a seqlock."""
from __future__ import annotations

import numpy as np
from multiprocessing import shared_memory

H, W = 480, 640
RGB_BYTES = H * W * 3
DEPTH_BYTES = H * W * 2
HDR = 8                      # int64 fields per slot: seq, t_host_ns, ts_rgb, ts_depth, skew_us, flags, host_skew_us, -
HDR_BYTES = HDR * 8
PREFIX = "kinect3dgs-"


class Ring:
    def __init__(self, n: int = 8, name: str | None = None, create: bool = False):
        self.n = n
        size = n * (RGB_BYTES + DEPTH_BYTES + HDR_BYTES)
        # Attached children share the parent's resource tracker (spawn), so only the creator unlinks.
        self.shm = shared_memory.SharedMemory(name=name, create=create, size=size if create else 0)
        buf = self.shm.buf
        self.rgb = np.ndarray((n, H, W, 3), np.uint8, buffer=buf, offset=0)
        self.depth = np.ndarray((n, H, W), np.uint16, buffer=buf, offset=n * RGB_BYTES)
        self.hdr = np.ndarray((n, HDR), np.int64, buffer=buf, offset=n * (RGB_BYTES + DEPTH_BYTES))
        if create:
            self.hdr[:, 0] = -1

    @property
    def name(self) -> str:
        return self.shm.name

    def reset(self) -> None:
        self.hdr[:, 0] = -1

    def publish(self, seq: int, rgb, depth, t_host_ns: int, ts_rgb: int, ts_depth: int,
                skew_ms: float, flags: int = 0, host_skew_ms: float = 0.0) -> int:
        s = seq % self.n
        self.hdr[s, 0] = -1                       # in flight
        self.rgb[s] = rgb
        self.depth[s] = depth
        self.hdr[s, 1:7] = (t_host_ns, ts_rgb, ts_depth, int(skew_ms * 1000), flags, int(host_skew_ms * 1000))
        self.hdr[s, 0] = seq                      # publish last
        return s

    def read(self, slot: int, seq: int):
        """Copy a slot out; None if the producer overwrote it meanwhile."""
        if self.hdr[slot, 0] != seq:
            return None
        rgb, depth, h = self.rgb[slot].copy(), self.depth[slot].copy(), self.hdr[slot].copy()
        if self.hdr[slot, 0] != seq or h[0] != seq:
            return None
        return rgb, depth, h

    def close(self) -> None:
        # drop numpy views first, otherwise SharedMemory.close() raises BufferError
        self.rgb = self.depth = self.hdr = None
        try:
            self.shm.close()
        except BufferError:
            pass

    def unlink(self) -> None:
        try:
            self.shm.unlink()
        except FileNotFoundError:
            pass
