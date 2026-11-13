"""Fake freenect for hardware-free tests. Behaviour via env: FAKE_FPS, FAKE_FRAMES, FAKE_SKEW_MS, FAKE_MODE (ok|stall|die|noopen), FAKE_FLAG_RET."""
import os
import time

import numpy as np

RESOLUTION_MEDIUM = 1
DEPTH_REGISTERED, DEPTH_MM, VIDEO_RGB = 4, 5, 0
FLAG_AUTO_EXPOSURE, FLAG_AUTO_WHITE_BALANCE, FLAG_AUTO_FLICKER, FLAG_OFF = 1 << 14, 1 << 1, 1 << 7, 0


class Kill(Exception):
    pass


class _Dev:
    ctx = None


def init():
    return object()


def open_device(ctx, index):
    if os.environ.get("FAKE_MODE") == "noopen":
        return None
    return _Dev()


def set_depth_mode(dev, res, mode):
    return 0


def set_video_mode(dev, res, mode):
    return 0


def set_flag(dev, flag, value):
    return int(os.environ.get("FAKE_FLAG_RET", "0"))


def close_device(dev):
    pass


def shutdown(ctx):
    pass


def runloop(depth=None, video=None, body=None, dev=None):
    fps = float(os.environ.get("FAKE_FPS", "100"))
    n_max = int(os.environ.get("FAKE_FRAMES", "0"))
    skew = float(os.environ.get("FAKE_SKEW_MS", "5")) / 1000.0
    mode = os.environ.get("FAKE_MODE", "ok")
    rgb = np.zeros((480, 640, 3), np.uint8)       # reused buffers: a missing copy shows up as aliasing
    dep = np.zeros((480, 640), np.uint16)
    i = 0
    try:
        while True:
            time.sleep(1.0 / fps)
            if not (mode == "stall" and i >= 20) and (n_max == 0 or i < n_max):
                rgb[:] = i % 256
                rgb[0, 0, 0] = i % 256
                video(dev, rgb, i * 5_992_740)
                time.sleep(skew)
                dep[:] = 1000 + (i % 3000)
                dep[0, 0] = i + 1
                depth(dev, dep, i * 5_992_740 + int(skew * 1000 * 179_782))
                i += 1
            if mode == "die" and i >= 20:
                os._exit(1)
            body(dev, None)
    except Kill:
        pass
