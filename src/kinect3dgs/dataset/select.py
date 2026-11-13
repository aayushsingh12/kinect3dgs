"""Frame selection (pure functions over session metadata; FR-2)."""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median


@dataclass
class Pick:
    session: str
    idx: int
    sharpness: float
    name: str


@dataclass
class Selection:
    picks: list[Pick]
    skipped: list[tuple[str, int, str]] = field(default_factory=list)   # (session, window start, reason)
    per_session: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def load_rows(session_dir: Path) -> list[dict]:
    lines = (session_dir / "frames.jsonl").read_text().splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def select_frames(sessions: dict[str, list[dict]], target: int = 200, min_frac: float = 0.6,
                  min_depth_valid: float = 0.05) -> Selection:
    """sessions: name -> rows (frames.jsonl). Equal share of windows per session; sharpest frame per window."""
    sel = Selection(picks=[])
    per = max(20, target // max(1, len(sessions)))
    for name, rows in sessions.items():
        if not rows:
            sel.warnings.append(f"session {name} has no frames")
            continue
        med = median(r["sharpness"] for r in rows)
        n_win = min(per, len(rows))
        bounds = [round(i * len(rows) / n_win) for i in range(n_win + 1)]
        kept = 0
        for a, b in zip(bounds[:-1], bounds[1:]):
            win = rows[a:b]
            best = max(win, key=lambda r: r["sharpness"])
            if best["sharpness"] < min_frac * med:
                sel.skipped.append((name, a, "blurry"))
                continue
            if best.get("depth_valid_frac", 1.0) < min_depth_valid:
                sel.skipped.append((name, a, "no depth"))
                continue
            sel.picks.append(Pick(name, best["idx"], float(best["sharpness"]), f"{name}_{best['idx']:06d}.png"))
            kept += 1
        sel.per_session[name] = {"frames": len(rows), "picked": kept, "median_sharpness": float(med)}
        if kept < 20:
            sel.warnings.append(f"session {name}: only {kept} usable frames (too short or mostly blurry?)")
    return sel


def materialise(sel: Selection, session_dirs: dict[str, Path], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for p in sel.picks:
        shutil.copyfile(session_dirs[p.session] / "rgb" / f"{p.idx:06d}.png", dest / p.name)
