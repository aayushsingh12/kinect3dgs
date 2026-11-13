"""Scan plans as data (FR-3). Edit text and numbers here; no UI code depends on them."""
from __future__ import annotations

from dataclasses import dataclass, field

# Full-resolution image shift in px per camera frame. Tuned on table01-03 with tools/tune_coach.py (2697 frame pairs):
# blurry frames (sharpness < 50 % of the rolling median) are 11 % of frames below 2 px/frame, 34 % at 4-6, ~40 % at
# 6-17 and above 50 % beyond ~17. 8 warns before most blur happens and fires on roughly the fastest 10 % of a normal scan.
SPEED_LIMIT_PX = 8.0


@dataclass(frozen=True)
class Zone:
    ideal: tuple          # metres (lo, hi): no hint inside
    warn: tuple           # metres (lo, hi): outside -> hard hint


@dataclass(frozen=True)
class Pass:
    title: str
    text: str
    seconds: int
    zone: Zone
    speed_limit_px: float = SPEED_LIMIT_PX


@dataclass(frozen=True)
class Plan:
    id: str
    title: str
    passes: tuple = field(default_factory=tuple)

    def public(self) -> dict:
        return {"id": self.id, "title": self.title, "passes": [
            {"title": p.title, "text": p.text, "seconds": p.seconds, "ideal": list(p.zone.ideal), "warn": list(p.zone.warn)}
            for p in self.passes]}


_TABLE_ZONE = Zone(ideal=(0.8, 1.5), warn=(0.6, 2.2))
_ROOM_ZONE = Zone(ideal=(1.0, 3.5), warn=(0.7, 4.2))

PLANS = {
    "table": Plan("table", "table / object", (
        Pass("pass 1", "walk slowly around the table, camera at chest height, looking at the objects", 90, _TABLE_ZONE),
        Pass("pass 2", "same loop, camera lower, looking slightly up", 90, _TABLE_ZONE),
        Pass("pass 3", "same loop, camera above, tilted down 60-70 degrees", 90, _TABLE_ZONE),
    )),
    "room": Plan("room", "room", (
        Pass("pass 1", "walk the edge of the room looking inward, camera at chest height", 180, _ROOM_ZONE),
        Pass("pass 2", "same loop at a different height, slightly tilted", 180, _ROOM_ZONE),
        Pass("pass 3", "face each wall from 1.5-2 m, then walk to the next", 180, _ROOM_ZONE),
    )),
}


def get_plan(plan_id: str) -> Plan:
    if plan_id not in PLANS:
        raise KeyError(f"unknown plan '{plan_id}'")
    return PLANS[plan_id]
