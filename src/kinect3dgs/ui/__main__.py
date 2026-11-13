"""python -m kinect3dgs.ui [--port 8770] [--demo] [--open]"""
from __future__ import annotations

import argparse
import signal
import threading
import webbrowser

try:                      # pycolmap installs its own SIGTERM crash handler on import: import it BEFORE we set ours
    import pycolmap  # noqa: F401
except ImportError:
    pass

from .controller import Controller
from .server import UiServer


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="kinect3dgs.ui", description=__doc__)
    ap.add_argument("--port", type=int, default=8770, help="0 = pick a free port")
    ap.add_argument("--demo", action="store_true", help="use a synthetic camera (no Kinect needed)")
    ap.add_argument("--open", action="store_true", help="open the page in your browser")
    a = ap.parse_args(argv)
    ctrl = Controller(demo=a.demo)
    srv = UiServer(ctrl, a.port)
    ctrl.start()
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"kinect3dgs ui: {srv.url}" + ("  (demo)" if a.demo else ""), flush=True)
    if a.open:
        webbrowser.open(srv.url)
    stop.wait()
    print("stopping ...", flush=True)
    ctrl.shutdown()
    srv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
