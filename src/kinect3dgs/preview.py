"""Loopback-only MJPEG preview of the latest RGB and depth (stdlib http.server + OpenCV encoding)."""
from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

HOST = "127.0.0.1"
PAGE = b"""<!doctype html><meta charset=utf-8><title>kinect3dgs preview</title>
<body style="margin:0;background:#111;color:#ccc;font:14px system-ui">
<div style="display:flex;flex-wrap:wrap;gap:8px;padding:8px">
<div><div>RGB</div><img src="/rgb.mjpg" width=640></div>
<div><div>Depth (registered, turbo, 0 = black)</div><img src="/depth.mjpg" width=640></div></div></body>"""


def depth_colormap(depth: np.ndarray, lo: int = 400, hi: int = 5000) -> np.ndarray:
    v = depth > 0
    norm = np.clip((depth.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    out = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    out[~v] = 0
    return out


class Preview:
    def __init__(self, port: int = 8765, max_fps: float = 10.0):
        self.port, self.min_dt = port, 1.0 / max_fps
        self._jpg = {"rgb": None, "depth": None}
        self._ver = 0
        self._last = 0.0
        self._cond = threading.Condition()
        self.server: ThreadingHTTPServer | None = None

    def update(self, rgb: np.ndarray, depth: np.ndarray) -> None:
        now = time.monotonic()
        if now - self._last < self.min_dt:
            return
        self._last = now
        ok1, a = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
        ok2, b = cv2.imencode(".jpg", depth_colormap(depth), [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok1 and ok2:
            with self._cond:
                self._jpg["rgb"], self._jpg["depth"] = a.tobytes(), b.tobytes()
                self._ver += 1
                self._cond.notify_all()

    def start(self) -> str:
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(PAGE)))
                    self.end_headers()
                    self.wfile.write(PAGE)
                elif self.path in ("/rgb.mjpg", "/depth.mjpg"):
                    key = "rgb" if self.path.startswith("/rgb") else "depth"
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    seen = -1
                    try:
                        while True:
                            with outer._cond:
                                outer._cond.wait_for(lambda: outer._ver != seen, timeout=2.0)
                                seen, data = outer._ver, outer._jpg[key]
                            if data is None:
                                continue
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(data))
                            self.wfile.write(data + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        return
                else:
                    self.send_error(404)

        self.server = ThreadingHTTPServer((HOST, self.port), H)   # loopback only, by construction
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return f"http://{HOST}:{self.server.server_address[1]}/"

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
