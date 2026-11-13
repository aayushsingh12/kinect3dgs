"""Local HTTP server: page, MJPEG video, SSE events, JSON API. Loopback only, token + Host + Origin checks (FR-11)."""
from __future__ import annotations

import json
import queue
import secrets
import shutil
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .controller import Controller
from .security import LOOPBACK, assert_loopback, check_request

STATIC = Path(__file__).parent / "static" / "index.html"
MAX_BODY = 10_000
MAX_VIDEO_FPS = 12.0


class UiServer:
    def __init__(self, controller: Controller, port: int = 8770, host: str = LOOPBACK):
        assert_loopback(host)
        self.ctrl = controller
        self.token = secrets.token_urlsafe(24)
        self.nonce = secrets.token_urlsafe(12)
        self.httpd = ThreadingHTTPServer((host, port), self._handler())
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.httpd.RequestHandlerClass.port = self.port

    @property
    def url(self) -> str:
        return f"http://{LOOPBACK}:{self.port}/"

    def serve_forever(self) -> None:
        self._serving = True
        self.httpd.serve_forever()

    def close(self) -> None:
        if getattr(self, "_serving", False):          # shutdown() blocks forever if serve_forever never ran
            self.httpd.shutdown()
        self.httpd.server_close()

    def _handler(self):
        srv = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            port = 0

            def log_message(self, *a):
                pass

            # -------- helpers
            def _reject(self, status: int) -> None:
                body = json.dumps({"ok": False, "error": {403: "forbidden", 415: "content type must be application/json",
                                                            404: "not found", 400: "bad request"}.get(status, "error")}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def _json(self, obj: dict, status: int = 200) -> None:
                body = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _check(self, need_token: bool = True, method: str = "GET") -> bool:
                status = check_request(method, self.path, self.headers, srv.token, self.port, need_token)
                if status:
                    self._reject(status)
                    return False
                return True

            # -------- GET
            def do_GET(self):
                path = urlparse(self.path).path
                if path in ("/", "/index.html"):
                    if not self._check(need_token=False):
                        return
                    html = STATIC.read_text().replace("__TOKEN__", srv.token).replace("__NONCE__", srv.nonce).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.send_header("Referrer-Policy", "no-referrer")
                    self.send_header("Content-Security-Policy",
                                     f"default-src 'none'; img-src 'self'; connect-src 'self'; "
                                     f"style-src 'nonce-{srv.nonce}'; script-src 'nonce-{srv.nonce}'; frame-ancestors 'none'")
                    self.send_header("Content-Length", str(len(html)))
                    self.end_headers()
                    self.wfile.write(html)
                elif path == "/api/state":
                    if self._check():
                        self._json(srv.ctrl.snapshot())
                elif path == "/video":
                    if self._check():
                        self._video(parse_qs(urlparse(self.path).query).get("view", ["rgb"])[0])
                elif path == "/events":
                    if self._check():
                        self._events()
                else:
                    self._reject(404)

            def _video(self, view: str) -> None:
                view = "depth" if view == "depth" else "rgb"
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    while not srv.ctrl.stop_evt.is_set():
                        t0 = time.monotonic()
                        jpg = srv.ctrl.jpeg(view)
                        if jpg:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg))
                            self.wfile.write(jpg + b"\r\n")
                        time.sleep(max(0.02, 1 / MAX_VIDEO_FPS - (time.monotonic() - t0)))
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return

            def _events(self) -> None:
                q = srv.ctrl.hub.subscribe()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    self.wfile.write(f"event: state\ndata: {json.dumps(srv.ctrl.snapshot())}\n\n".encode())
                    self.wfile.flush()
                    while not srv.ctrl.stop_evt.is_set():
                        try:
                            kind, data = q.get(timeout=10)
                            self.wfile.write(f"event: {kind}\ndata: {json.dumps(data)}\n\n".encode())
                        except queue.Empty:
                            self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    srv.ctrl.hub.unsubscribe(q)

            # -------- POST
            def do_POST(self):
                if not self._check(method="POST"):
                    return
                path = urlparse(self.path).path
                try:
                    n = int(self.headers.get("Content-Length", "0"))
                    if n > MAX_BODY:
                        return self._reject(400)
                    body = json.loads(self.rfile.read(n) or b"{}")
                    if not isinstance(body, dict):
                        raise ValueError
                except ValueError:
                    return self._reject(400)
                ctrl = srv.ctrl
                routes = {
                    "/api/check": lambda: ctrl.submit("check"),
                    "/api/scan/start": lambda: ctrl.submit("start_scan", name=str(body.get("name", "")),
                                                           plan=str(body.get("plan", "")), out=str(body.get("out", ""))),
                    "/api/scan/resume": lambda: ctrl.submit("resume"),
                    "/api/scan/pause": lambda: ctrl.submit("pause"),
                    "/api/scan/next": lambda: ctrl.submit("next_pass", force=bool(body.get("force", False))),
                    "/api/scan/finish": lambda: ctrl.submit("finish"),
                    "/api/process/start": lambda: ctrl.submit("process"),
                    "/api/process/cancel": lambda: ctrl.submit("cancel"),
                    "/api/reset": lambda: ctrl.submit("reset"),
                }
                if path == "/api/validate-out":
                    ok, err = ctrl.validate_out(str(body.get("out", "")), str(body.get("name", "")), str(body.get("plan", "table")))
                    return self._json({"ok": ok, "error": err})
                if path == "/api/open-folder":
                    out = ctrl.last_out
                    if ctrl.state != "done" or out is None or not out.exists() or not shutil.which("xdg-open"):
                        return self._json({"ok": False, "error": "nothing to open"}, 409)
                    subprocess.Popen(["xdg-open", str(out)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return self._json({"ok": True})
                if path not in routes:
                    return self._reject(404)
                ok, err = routes[path]()
                self._json({"ok": ok, "error": err, "state": ctrl.state}, 200 if ok else 409)

        return H
