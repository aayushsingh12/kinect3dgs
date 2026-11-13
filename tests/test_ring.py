import multiprocessing as mp
import os

import numpy as np

from kinect3dgs.ring import PREFIX, Ring


def _child(name, q):
    r = Ring(4, name)
    q.put((int(r.rgb[1, 0, 0, 0]), int(r.depth[1, 0, 0]), int(r.hdr[1, 0])))
    r.close()


def make(n=4):
    return Ring(n, name=f"{PREFIX}test-{os.getpid()}-{os.urandom(2).hex()}", create=True)


def test_publish_read_roundtrip():
    r = make()
    rgb = np.full((480, 640, 3), 7, np.uint8)
    d = np.full((480, 640), 900, np.uint16)
    s = r.publish(5, rgb, d, 123, 1, 2, 4.5)
    a, b, h = r.read(s, 5)
    assert np.array_equal(a, rgb) and np.array_equal(b, d) and h[1] == 123 and h[4] == 4500
    r.close(); r.unlink()


def test_overwritten_slot_is_detected():
    r = make(4)
    z = np.zeros((480, 640, 3), np.uint8), np.zeros((480, 640), np.uint16)
    s = r.publish(1, *z, 0, 0, 0, 0)
    r.publish(5, *z, 0, 0, 0, 0)          # same slot (5 % 4 == 1)
    assert r.read(s, 1) is None
    assert r.read(s, 5) is not None
    r.close(); r.unlink()


def test_two_process_visibility():
    r = make(4)
    rgb = np.full((480, 640, 3), 42, np.uint8)
    d = np.full((480, 640), 4242, np.uint16)
    r.publish(1, rgb, d, 0, 0, 0, 0)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(r.name, q))
    p.start()
    assert q.get(timeout=20) == (42, 4242, 1)
    p.join()
    r.close(); r.unlink()
    assert not os.path.exists("/dev/shm/" + r.name)


def test_unlink_idempotent():
    r = make()
    r.close(); r.unlink(); r.unlink()
