import json
import socket
import urllib.error
import urllib.request

import pytest

from kinect3dgs.ui.controller import Controller
from kinect3dgs.ui.security import allowed_hosts, assert_loopback, check_request
from kinect3dgs.ui.server import UiServer

PORT = 8770
TOKEN = "tok"


class H(dict):
    def get(self, k, d=None):
        return super().get(k, d)


def chk(method="GET", path="/api/state", headers=None, token_needed=True):
    h = {"Host": f"127.0.0.1:{PORT}", "X-Token": TOKEN}
    h.update(headers or {})
    return check_request(method, path, H(h), TOKEN, PORT, token_needed)


def test_matrix():
    assert chk() is None
    assert chk(headers={"X-Token": "wrong"}) == 403
    assert chk(headers={"X-Token": ""}) == 403
    assert chk(headers={"Host": "evil.example:8770"}) == 403
    assert chk(headers={"Host": "127.0.0.1:9999"}) == 403
    assert chk(headers={"Host": f"localhost:{PORT}"}) is None
    assert chk(headers={"Origin": "http://evil.example"}) == 403
    assert chk(headers={"Origin": f"http://127.0.0.1:{PORT}"}) is None
    assert chk(path=f"/events?t={TOKEN}", headers={"X-Token": ""}) is None          # query token (EventSource/img)
    assert chk(path="/events?t=bad", headers={"X-Token": ""}) == 403
    assert chk(method="POST", headers={"Content-Type": "text/plain"}) == 415
    assert chk(method="POST", headers={"Content-Type": "application/json; charset=utf-8"}) is None
    assert chk(path="/", headers={"X-Token": ""}, token_needed=False) is None


def test_bind_must_be_loopback():
    assert_loopback("127.0.0.1")
    for bad in ("0.0.0.0", "192.168.1.5", "localhost", ""):
        with pytest.raises(ValueError):
            assert_loopback(bad)
    with pytest.raises(ValueError):
        UiServer(Controller(demo=True), 0, host="0.0.0.0")


@pytest.fixture
def server():
    c = Controller(demo=True)
    s = UiServer(c, 0)
    import threading
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield s
    s.close()


def req(s, path, method="GET", headers=None, body=None, host=None):
    h = {"Host": host or f"127.0.0.1:{s.port}"}
    h.update(headers or {})
    r = urllib.request.Request(f"http://127.0.0.1:{s.port}{path}", method=method, headers=h, data=body)
    try:
        resp = urllib.request.urlopen(r, timeout=5)
        return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def test_http_security_matrix(server):
    s = server
    code, hdr, body = req(s, "/")
    assert code == 200 and b"const T = " in body
    assert not any(k.lower().startswith("access-control") for k in hdr)
    csp = hdr["Content-Security-Policy"]
    assert "default-src 'none'" in csp and "'unsafe-inline'" not in csp and "frame-ancestors 'none'" in csp
    assert req(s, "/", host="evil.example:80")[0] == 403
    assert req(s, "/api/state")[0] == 403                                               # no token
    tok = body.decode().split('const T = "')[1].split('"')[0]
    assert req(s, "/api/state", headers={"X-Token": tok})[0] == 200
    assert req(s, "/api/state", headers={"X-Token": tok, "Origin": "http://evil.example"})[0] == 403
    assert req(s, "/api/reset", "POST", {"X-Token": tok}, b"{}")[0] == 415                # no JSON content type
    assert req(s, "/api/reset", "POST", {"X-Token": "bad", "Content-Type": "application/json"}, b"{}")[0] == 403
    assert req(s, f"/events?t=nope")[0] == 403 and req(s, "/video?t=nope")[0] == 403
    assert req(s, "/nope", headers={"X-Token": tok})[0] == 404
    assert req(s, "/api/open-folder", "POST", {"X-Token": tok, "Content-Type": "application/json"}, b"{}")[0] == 409  # nothing to open
    big = b'{"a":"' + b"x" * 20000 + b'"}'
    assert req(s, "/api/check", "POST", {"X-Token": tok, "Content-Type": "application/json"}, big)[0] == 400
    assert req(s, "/api/check", "POST", {"X-Token": tok, "Content-Type": "application/json"}, b"not json")[0] == 400


def test_token_changes_per_server(server):
    c2 = Controller(demo=True)
    s2 = UiServer(c2, 0)
    try:
        assert s2.token != server.token and s2.nonce != server.nonce
    finally:
        s2.close()


def test_listens_only_on_loopback(server):
    assert server.httpd.server_address[0] == "127.0.0.1"
    # a connection to the machine's non-loopback address must not reach the server
    try:
        ext = socket.gethostbyname(socket.gethostname())
    except OSError:
        pytest.skip("no external address")
    if ext.startswith("127."):
        pytest.skip("hostname resolves to loopback")
    s = socket.socket()
    s.settimeout(1)
    assert s.connect_ex((ext, server.port)) != 0
    s.close()
