import threading
import time

import numpy as np

from kinect3dgs.writer import PairWriter


def row(i):
    return {"idx": i}


def test_backpressure_drops_and_counts(tmp_path):
    w = PairWriter(tmp_path, threads=1, queue_max=2)
    gate = threading.Event()
    orig = w._encode
    w._encode = lambda *a: (gate.wait(5), orig(*a))[1]
    rgb, d = np.zeros((480, 640, 3), np.uint8), np.zeros((480, 640), np.uint16)
    results = [w.submit(i, rgb, d, row(i)) for i in range(5)]
    assert results == [True, True, False, False, False] and w.dropped == 3
    gate.set()
    w.close()
    assert w.written == 2


def test_jsonl_order_matches_index(tmp_path):
    w = PairWriter(tmp_path, threads=3)
    rgb, d = np.zeros((480, 640, 3), np.uint8), np.zeros((480, 640), np.uint16)
    for i in range(20):
        w.submit(i, rgb, d, row(i))
    w.close()
    import json
    idx = [json.loads(l)["idx"] for l in (tmp_path / "frames.jsonl").read_text().splitlines()]
    assert idx == list(range(20))
