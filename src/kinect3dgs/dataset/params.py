"""Build parameters and path-safety rules (FR-1, D8, D9)."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .errors import UsageError

REPO = Path(__file__).resolve().parents[3]
SESSIONS_DIR = REPO / "data" / "sessions"
FORBIDDEN_ROOTS = [REPO / d for d in ("src", "tests", "scripts", "specs", "patches", "pywrapper", ".git")]
MB_PER_FRAME = 1.0          # undistorted PNG + work copies, generous


@dataclass
class BuildParams:
    sessions: list[Path]
    out: Path
    name: str
    work_dir: Path
    target_frames: int = 200
    min_sharpness_frac: float = 0.6
    max_image_size: int = 800
    matching: str = "auto"              # auto | exhaustive | sequential
    metric_scale: bool = True
    text: bool = False
    force: bool = False
    clean_work: bool = False
    quiet: bool = False
    progress_jsonl: bool = False
    extra: dict = field(default_factory=dict)


def resolve_session(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if os.sep not in name_or_path and not p.exists():
        p = SESSIONS_DIR / name_or_path
    p = p.expanduser().resolve()
    if not (p / "session.json").exists():
        raise UsageError(f"unknown session '{name_or_path}' (no session.json at {p})")
    return p


def _inside(p: Path, root: Path) -> bool:
    try:
        p.relative_to(root)
        return True
    except ValueError:
        return False


def check_paths(params: BuildParams) -> None:
    out = params.out.expanduser().resolve()
    work = params.work_dir.expanduser().resolve()
    params.out, params.work_dir = out, work
    if out in (Path("/"), Path.home().resolve()):
        raise UsageError(f"refusing to use {out} as an output folder")
    for root in FORBIDDEN_ROOTS:
        if _inside(out, root.resolve()) or _inside(work, root.resolve()):
            raise UsageError(f"refusing to write inside the project folder {root.name}/")
    for s in params.sessions:
        s = s.resolve()
        if out == s or _inside(s, out) or _inside(out, s):
            raise UsageError(f"--out {out} overlaps the session folder {s}")
    if _inside(work, out) or _inside(out, work):
        raise UsageError("--work-dir and --out must not contain each other")
    if out.exists():
        if not out.is_dir():
            raise UsageError(f"{out} exists and is not a folder")
        if any(out.iterdir()):
            if not params.force:
                raise UsageError(f"output folder {out} is not empty; use --force to replace an existing dataset")
            if not (out / "dataset.json").exists():
                raise UsageError(f"refusing to replace {out}: it is not a dataset (no dataset.json)")
    parent = out.parent
    if not parent.exists():
        raise UsageError(f"parent folder {parent} does not exist")
    if not os.access(parent, os.W_OK):
        raise UsageError(f"cannot write to {parent}")
    need = int(params.target_frames * MB_PER_FRAME * 1e6 * 3)
    free = shutil.disk_usage(parent).free
    if free < max(need, 500 * 1024**2):
        raise UsageError(f"not enough free disk space in {parent} ({free / 1e9:.1f} GB free)")
