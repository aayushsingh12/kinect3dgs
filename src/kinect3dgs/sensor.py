"""Kinect access via libfreenect's sync API, isolated in a child process.

sync_get_* blocks in C and cannot be interrupted, so every capture runs in a
spawned process that is killed on timeout. Arrays are copied out of
libfreenect's buffer (the wrapper returns views that the next call overwrites).
"""
from __future__ import annotations

import fcntl
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import KinectBusyError, KinectNotFoundError, KinectTimeoutError

SYSFS_USB = Path(os.environ.get("KINECT3DGS_SYSFS_USB", "/sys/bus/usb/devices"))
PROC_MODULES = Path(os.environ.get("KINECT3DGS_PROC_MODULES", "/proc/modules"))
CAMERA_PIDS = {"02ae", "02bf"}
L0_HINT = ("No Kinect camera found. Check: 12 V adapter plugged in at both ends, USB cable, "
           "try a USB 2.0 port; run scripts/usb_list.sh (expect 045e:02ae).")


@dataclass
class UsbDevice:
    bus_id: str
    vid: str
    pid: str
    product: str


@dataclass
class Capture:
    rgb: np.ndarray
    depth_mm: np.ndarray
    ts_rgb: int
    ts_depth: int
    stamps: list
    seconds: float


def list_devices() -> list[UsbDevice]:
    out = []
    if not SYSFS_USB.exists():
        return out
    for d in sorted(SYSFS_USB.iterdir()):
        try:
            vid = (d / "idVendor").read_text().strip()
            pid = (d / "idProduct").read_text().strip()
        except OSError:
            continue
        if vid != "045e":
            continue
        try:
            product = (d / "product").read_text().strip()
        except OSError:
            product = ""
        out.append(UsbDevice(d.name, vid, pid, product))
    return out


USBDEVFS_RESET = 0x5514


def reset_camera(wait: float = 20.0) -> bool:
    """USB-reset the Kinect camera (clears a desynced control channel) and wait for it to re-enumerate.

    Needs write access to /dev/bus/usb/... which the udev rule (MODE=0666) provides.
    """
    for d in SYSFS_USB.iterdir() if SYSFS_USB.exists() else []:
        try:
            if (d / "idVendor").read_text().strip() != "045e" or (d / "idProduct").read_text().strip() not in CAMERA_PIDS:
                continue
            node = f"/dev/bus/usb/{int((d / 'busnum').read_text()):03d}/{int((d / 'devnum').read_text()):03d}"
            fd = os.open(node, os.O_WRONLY)
            try:
                fcntl.ioctl(fd, USBDEVFS_RESET, 0)
            finally:
                os.close(fd)
        except (OSError, ValueError):
            return False
        break
    else:
        return False
    deadline = time.monotonic() + wait
    time.sleep(2.0)
    while time.monotonic() < deadline:
        if any(x.pid in CAMERA_PIDS for x in list_devices()):
            time.sleep(1.5)
            return True
        time.sleep(0.5)
    return False


def _gspca_loaded() -> bool:
    try:
        return any(line.startswith("gspca_kinect ") for line in PROC_MODULES.read_text().splitlines())
    except OSError:
        return False


def _device_count() -> int:
    import freenect
    ctx = freenect.init()
    if ctx is None:
        raise KinectNotFoundError("freenect_init failed", L0_HINT)
    try:
        return freenect.num_devices(ctx)
    finally:
        freenect.shutdown(ctx)


def _worker(conn, warmup: int, fps_seconds: float) -> None:
    import freenect
    try:
        for _ in range(warmup):
            if freenect.sync_get_video() is None or freenect.sync_get_depth(format=freenect.DEPTH_MM) is None:
                conn.send(("error", "open"))
                return
        v = freenect.sync_get_video(format=freenect.VIDEO_RGB)
        d = freenect.sync_get_depth(format=freenect.DEPTH_MM)
        if v is None or d is None:
            conn.send(("error", "open"))
            return
        rgb, ts_v = np.array(v[0], copy=True), int(v[1])
        depth, ts_d = np.array(d[0], copy=True), int(d[1])
        stamps, t0 = [], time.monotonic()
        while time.monotonic() - t0 < fps_seconds:
            f = freenect.sync_get_video()
            if f is None:
                conn.send(("error", "stream"))
                return
            stamps.append(int(f[1]))
        conn.send(("ok", rgb, depth, ts_v, ts_d, stamps, time.monotonic() - t0 if fps_seconds else 0.0))
    finally:
        freenect.sync_stop()
        conn.close()


def _capture_once(warmup, fps_seconds, timeout, _target, _check_devices) -> Capture:
    if _check_devices and _device_count() == 0:
        raise KinectNotFoundError("no Kinect device seen by libfreenect", L0_HINT)
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_target, args=(child, warmup, fps_seconds), daemon=True)
    p.start()
    child.close()
    try:
        if not parent.poll(timeout + fps_seconds):
            raise KinectTimeoutError(f"no frames within {timeout + fps_seconds:.0f}s (device silent?)")
        try:
            msg = parent.recv()
        except EOFError:
            raise KinectNotFoundError("capture process died", L0_HINT) from None
    finally:
        if p.is_alive():
            p.kill()
        p.join()
    if msg[0] == "error":
        if _gspca_loaded():
            raise KinectBusyError("gspca_kinect holds the camera; run scripts/system_setup.sh")
        raise KinectNotFoundError(
            f"could not {msg[1]} the device (permissions? run scripts/system_setup.sh --check)", L0_HINT)
    return Capture(*msg[1:])


def capture(warmup: int = 15, fps_seconds: float = 3.0, timeout: float = 30.0, retries: int = 2,
            _target=_worker, _check_devices: bool = True, _reset=reset_camera) -> Capture:
    """Capture with automatic recovery: on an open failure or silent device, USB-reset the camera and retry."""
    for attempt in range(retries + 1):
        try:
            return _capture_once(warmup, fps_seconds, timeout, _target, _check_devices)
        except (KinectNotFoundError, KinectTimeoutError):
            if attempt == retries:
                raise
            _reset()
        # KinectBusyError (gspca) is not retried: it needs scripts/system_setup.sh.


def grab_pair(warmup: int = 15, timeout: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    c = capture(warmup=warmup, fps_seconds=0.0, timeout=timeout)
    return c.rgb, c.depth_mm


def grab_rgb(**kw) -> np.ndarray:
    return grab_pair(**kw)[0]


def grab_depth_mm(**kw) -> np.ndarray:
    return grab_pair(**kw)[1]
