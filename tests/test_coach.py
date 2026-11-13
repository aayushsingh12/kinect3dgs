import numpy as np
import cv2

from kinect3dgs.ui.coach import Coach, HOLD_S
from kinect3dgs.ui.plans import PLANS


def pp():
    return PLANS["table"].passes[0]


def textured(seed=0, bright=120):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8), (0, 0), 1.5)
    return np.clip(img.astype(int) * bright // 120, 0, 255).astype(np.uint8)


def depth(mm=1100, frac=1.0):
    d = np.full((480, 640), mm, np.uint16)
    d[:, int(640 * frac):] = 0
    return d


def run(coach, rgb, d, seconds, t0=0.0, fps=30):
    hint = None
    for i in range(int(seconds * fps)):
        hint = coach.update(rgb, d, t0 + i / fps)[1]
    return hint


def test_good_scene_has_no_hint():
    c = Coach(pp())
    assert run(c, textured(), depth(1100), 2) is None


def test_distance_rules_and_priority():
    c = Coach(pp())
    assert run(c, textured(), depth(500), 1.0) == "too close"
    c = Coach(pp())
    assert run(c, textured(), depth(2600), 1.0) == "too far"
    c = Coach(pp())
    assert "no depth" in run(c, textured(), depth(1100, frac=0.05), 1.0)      # outranks distance


def test_hysteresis_hold_and_release():
    c = Coach(pp())
    img = textured()
    assert run(c, img, depth(500), HOLD_S * 0.5) is None                      # not yet
    assert run(c, img, depth(500), 1.0, t0=1.0) == "too close"
    assert run(c, img, depth(1100), HOLD_S * 0.4, t0=3.0) == "too close"       # lingers briefly
    assert run(c, img, depth(1100), 1.5, t0=3.5) is None


def test_speed_rule_from_image_shift():
    c = Coach(pp())
    base = textured()
    hint = None
    for i in range(40):
        hint = c.update(np.roll(base, i * 40, axis=1), depth(1100), i / 30)[1]    # 40 px per frame
    assert hint in ("slow down", "blurry - slow down")
    c = Coach(pp())
    for i in range(60):
        hint = c.update(base, depth(1100), i / 30)[1]                              # standing still
    assert hint is None


def test_blur_is_relative_not_absolute():
    sharp_scene, flat_scene = textured(), cv2.GaussianBlur(textured(1), (0, 0), 6)
    for scene in (sharp_scene, flat_scene):                                   # both steady scenes: no blur hint
        c = Coach(pp())
        assert run(c, scene, depth(1100), 3) is None
    c = Coach(pp())
    run(c, sharp_scene, depth(1100), 3)
    hint = run(c, cv2.GaussianBlur(sharp_scene, (0, 0), 8), depth(1100), 1.0, t0=3.0)   # sudden loss of sharpness
    assert hint == "blurry - slow down"


def test_light_rules():
    c = Coach(pp())
    assert run(c, textured(bright=20), depth(1100), 1.0) == "too dark"
    c = Coach(pp())
    assert run(c, np.full((480, 640, 3), 255, np.uint8), depth(1100), 1.0) == "too bright"


def test_one_hint_at_a_time_priority():
    c = Coach(pp())
    h = run(c, np.full((480, 640, 3), 255, np.uint8), depth(500), 1.0)          # too close AND too bright
    assert h == "too close"


def test_room_zone_differs():
    c = Coach(PLANS["room"].passes[0])
    assert run(c, textured(), depth(2500), 1.0) is None                        # fine for a room, too far for a table
    c = Coach(PLANS["table"].passes[0])
    assert run(c, textured(), depth(2500), 1.0) == "too far"
