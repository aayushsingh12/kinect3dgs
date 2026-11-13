import numpy as np

from kinect3dgs.dataset.select import select_frames


def rows(n, sharp=None, valid=0.7):
    rng = np.random.default_rng(0)
    sharp = sharp if sharp is not None else rng.uniform(50, 150, n)
    return [{"idx": i, "sharpness": float(sharp[i]), "depth_valid_frac": valid} for i in range(n)]


def test_one_frame_per_window_and_sharpest():
    r = rows(240)
    r[5]["sharpness"] = 999.0
    sel = select_frames({"a": r}, target=20, min_frac=0.0)
    names = [p.idx for p in sel.picks]
    assert 5 in names and len(names) <= 21
    assert all(b > a for a, b in zip(names, names[1:]))


def test_equal_share_per_session():
    sel = select_frames({"a": rows(900), "b": rows(300), "c": rows(900)}, target=210, min_frac=0.0)
    cnt = {k: v["picked"] for k, v in sel.per_session.items()}
    assert abs(cnt["a"] - cnt["b"]) <= 2 and abs(cnt["a"] - cnt["c"]) <= 2


def test_blurry_windows_skipped_relative_to_session_median():
    sharp = np.full(120, 100.0)
    sharp[40:60] = 10.0            # a blurry stretch
    sel = select_frames({"a": rows(120, sharp)}, target=20, min_frac=0.6)
    assert any(r == "blurry" for _, _, r in sel.skipped)
    assert all(not (40 <= p.idx < 60) for p in sel.picks)


def test_no_depth_windows_skipped():
    sel = select_frames({"a": rows(120, valid=0.01)}, target=20)
    assert not sel.picks and all(r == "no depth" for _, _, r in sel.skipped)


def test_scene_content_does_not_change_relative_rule():
    low = select_frames({"a": rows(120, np.full(120, 5.0))}, target=20)
    high = select_frames({"a": rows(120, np.full(120, 500.0))}, target=20)
    assert len(low.picks) == len(high.picks)


def test_deterministic_and_short_session_warning():
    a = select_frames({"a": rows(30)}, target=20)
    b = select_frames({"a": rows(30)}, target=20)
    assert [p.name for p in a.picks] == [p.name for p in b.picks]
    assert any("only" in w for w in a.warnings)
