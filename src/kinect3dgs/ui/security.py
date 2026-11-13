"""Request checks for the local server (FR-11, D8): token, Host, Origin, content type."""
from __future__ import annotations

import hmac
from urllib.parse import parse_qs, urlparse

LOOPBACK = "127.0.0.1"


def allowed_hosts(port: int) -> set[str]:
    return {f"127.0.0.1:{port}", f"localhost:{port}"}


def allowed_origins(port: int) -> set[str]:
    return {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}


def check_request(method: str, path: str, headers, token: str, port: int, need_token: bool = True) -> int | None:
    """Return an HTTP status to reject with, or None when the request is acceptable."""
    if headers.get("Host") not in allowed_hosts(port):
        return 403
    origin = headers.get("Origin")
    if origin and origin not in allowed_origins(port):
        return 403
    if need_token:
        supplied = headers.get("X-Token") or (parse_qs(urlparse(path).query).get("t") or [""])[0]
        if not hmac.compare_digest(supplied.encode(), token.encode()):
            return 403
    if method == "POST" and headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
        return 415
    return None


def assert_loopback(host: str) -> None:
    if host != LOOPBACK:
        raise ValueError(f"the UI only binds {LOOPBACK}, not {host!r}")
