from kinect3dgs.pairing import Frame, Pairer, ticks_diff

MS = 1_000_000
TPM = 179_782            # device ticks per ms (measured: ~5.99M ticks per 33.3 ms frame)
PERIOD = 33.333


def rgb(i, host_off=0.0):
    return Frame(int((i * PERIOD + host_off) * MS), int(i * PERIOD * TPM) & 0xFFFFFFFF, None)


def dep(i, dev_off_ms=0.0, host_off=0.0):
    return Frame(int((i * PERIOD + host_off) * MS), int((i * PERIOD + dev_off_ms) * TPM) & 0xFFFFFFFF, None)


def run(n, dev_off=0.0, rgb_host=0.0, dep_host=0.0):
    p, out = Pairer(), []
    for i in range(n):
        out += p.push_rgb(rgb(i, rgb_host))
        out += p.push_depth(dep(i, dev_off, dep_host))
    return p, out


def test_matched_streams_one_to_one():
    p, out = run(100, dev_off=8.0)
    assert len(out) >= 98
    assert all(abs(x.skew_ms - 8.0) < 0.2 and not x.bad for x in out)


def test_host_arrival_phase_does_not_matter():
    """Regression: a 30 ms host-arrival offset (USB phase) must not change which frames pair."""
    _, out = run(100, dev_off=0.0, dep_host=30.0)
    assert len(out) >= 98 and all(abs(x.skew_ms) < 0.2 for x in out)
    assert abs(out[10].host_skew_ms - 30.0) < 0.5


def test_bad_flag_above_threshold():
    _, out = run(20, dev_off=16.0)
    assert out and all(x.bad for x in out)


def test_beyond_half_period_is_not_paired():
    p = Pairer()
    p.push_rgb(rgb(0)); p.push_rgb(rgb(1))
    assert p.push_depth(dep(5)) == []


def test_frames_used_at_most_once():
    _, out = run(60, dev_off=3.0)
    ids = set()
    for x in out:
        assert id(x.rgb) not in ids and id(x.depth) not in ids
        ids |= {id(x.rgb), id(x.depth)}


def test_one_stream_stalls_memory_bounded():
    p = Pairer(keep=3)
    for i in range(500):
        p.push_rgb(rgb(i))
    assert len(p.rgb) <= 3 and p.unmatched >= 497


def test_recovers_after_stall():
    p, out = Pairer(), []
    for i in range(10):
        p.push_rgb(rgb(i))
    for i in range(10, 25):
        out += p.push_rgb(rgb(i)); out += p.push_depth(dep(i, 4.0))
    assert len(out) >= 12


def test_device_counter_wraparound():
    p, out = Pairer(), []
    base = 2**32 - 10 * int(PERIOD * TPM)
    for i in range(30):
        out += p.push_rgb(Frame(0, (base + int(i * PERIOD * TPM)) & 0xFFFFFFFF, None))
        out += p.push_depth(Frame(0, (base + int((i * PERIOD + 2) * TPM)) & 0xFFFFFFFF, None))
    assert len(out) >= 28 and all(abs(x.skew_ms - 2.0) < 0.2 for x in out)
    assert ticks_diff(2**32 - 5, 5) == 10
