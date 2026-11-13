"""Session folder schema, atomic metadata writing and loading."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SCHEMA_VERSION = 1
REQUIRED_META = {"schema_version", "started_utc", "formats", "args", "stats", "intrinsics"}
REQUIRED_ROW = {"idx", "t_host_s", "ts_rgb", "ts_depth", "skew_ms", "sharpness", "depth_valid_frac", "bad_pair"}


class SessionError(ValueError):
    pass


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def rgb_path(root: Path, idx: int) -> Path:
    return root / "rgb" / f"{idx:06d}.png"


def depth_path(root: Path, idx: int) -> Path:
    return root / "depth" / f"{idx:06d}.png"


@dataclass
class Session:
    root: Path
    meta: dict
    rows: list

    def __len__(self) -> int:
        return len(self.rows)

    def frame(self, i: int):
        rgb = cv2.cvtColor(cv2.imread(str(rgb_path(self.root, i)), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(depth_path(self.root, i)), cv2.IMREAD_UNCHANGED)
        return rgb, depth, self.rows[i]

    def frames(self):
        for i in range(len(self.rows)):
            yield self.frame(i)


def load_session(path, check_images: bool = True) -> Session:
    root = Path(path)
    try:
        meta = json.loads((root / "session.json").read_text())
    except OSError as e:
        raise SessionError(f"missing session.json: {e}") from e
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise SessionError(f"unsupported schema_version {meta.get('schema_version')}")
    missing = REQUIRED_META - meta.keys()
    if missing:
        raise SessionError(f"session.json missing keys: {sorted(missing)}")
    rows = []
    jl = root / "frames.jsonl"
    if jl.exists():
        for line in jl.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    last_t = -1.0
    for i, r in enumerate(rows):
        if REQUIRED_ROW - r.keys():
            raise SessionError(f"row {i} missing keys {sorted(REQUIRED_ROW - r.keys())}")
        if r["idx"] != i:
            raise SessionError(f"non-contiguous idx at row {i}: {r['idx']}")
        if r["t_host_s"] < last_t:
            raise SessionError(f"t_host_s decreases at row {i}")
        last_t = r["t_host_s"]
        if not rgb_path(root, i).exists() or not depth_path(root, i).exists():
            raise SessionError(f"missing image files for idx {i}")
    s = Session(root, meta, rows)
    if check_images and rows:
        rgb, depth, _ = s.frame(len(rows) - 1)
        if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8 or depth.shape != (480, 640) or depth.dtype != np.uint16:
            raise SessionError(f"bad image shapes: rgb {rgb.shape} {rgb.dtype}, depth {depth.shape} {depth.dtype}")
    return s
