import pytest

from kinect3dgs.ui.plans import PLANS, SPEED_LIMIT_PX, get_plan


def test_plans_complete_and_public():
    assert set(PLANS) == {"table", "room"}
    for p in PLANS.values():
        assert len(p.passes) == 3
        pub = p.public()
        assert pub["id"] == p.id and len(pub["passes"]) == 3
        for ps in p.passes:
            assert ps.seconds in (90, 180) and ps.zone.warn[0] < ps.zone.ideal[0] < ps.zone.ideal[1] < ps.zone.warn[1]
            assert 0 < ps.speed_limit_px == SPEED_LIMIT_PX


def test_table_and_room_distances_and_durations():
    assert PLANS["table"].passes[0].zone.ideal == (0.8, 1.5) and PLANS["table"].passes[0].seconds == 90
    assert PLANS["room"].passes[0].zone.ideal == (1.0, 3.5) and PLANS["room"].passes[0].seconds == 180


def test_unknown_plan():
    with pytest.raises(KeyError):
        get_plan("nope")
