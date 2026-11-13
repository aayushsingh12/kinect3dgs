"""Validate a COLMAP dataset folder (FR-9)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pycolmap

MIN_POINTS = 3000


@dataclass
class Report:
    ok: bool
    checks: list = field(default_factory=list)       # (name, passed, detail)
    summary: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "summary": self.summary,
                "checks": [{"name": n, "passed": p, "detail": d} for n, p, d in self.checks]}


def validate_dataset(path, min_points: int = MIN_POINTS) -> Report:
    root = Path(path)
    checks = []

    def add(name, passed, detail=""):
        checks.append((name, bool(passed), detail))
        return passed

    sp = root / "sparse" / "0"
    files_ok = all((sp / f).exists() for f in ("cameras.bin", "images.bin", "points3D.bin"))
    add("sparse/0 files", files_ok, str(sp))
    add("images folder", (root / "images").is_dir())
    if not files_ok or not (root / "images").is_dir():
        return Report(False, checks)
    try:
        rec = pycolmap.Reconstruction(sp)
    except Exception as e:
        add("model loads", False, str(e))
        return Report(False, checks)
    add("model loads", True)
    cams = list(rec.cameras.values())
    add("camera model", all(c.model_name in ("PINHOLE", "SIMPLE_PINHOLE") for c in cams),
        ",".join(c.model_name for c in cams))
    missing, wrong_size, nan_pose, names = [], [], [], []
    for im in rec.images.values():
        if not im.has_pose:
            continue
        names.append(im.name)
        f = root / "images" / im.name
        if not f.exists():
            missing.append(im.name)
            continue
        cam = rec.cameras[im.camera_id]
        import cv2
        h, w = cv2.imread(str(f), cv2.IMREAD_UNCHANGED).shape[:2]
        if (w, h) != (int(cam.width), int(cam.height)):
            wrong_size.append(im.name)
        if not np.isfinite(im.cam_from_world().matrix()).all():
            nan_pose.append(im.name)
    add("registered images present", not missing, f"{len(missing)} missing" if missing else f"{len(names)} images")
    add("image sizes match camera", not wrong_size, f"{len(wrong_size)} mismatched" if wrong_size else "")
    add("poses finite", not nan_pose, f"{len(nan_pose)} bad" if nan_pose else "")
    add("unique names", len(names) == len(set(names)))
    add(f"points >= {min_points}", rec.num_points3D() >= min_points, str(rec.num_points3D()))
    dj = root / "dataset.json"
    if dj.exists():
        meta = json.loads(dj.read_text())
        add("dataset.json matches model", meta.get("sfm", {}).get("registered") == len(names),
            f"json {meta.get('sfm', {}).get('registered')} vs model {len(names)}")
    summary = {"registered": len(names), "points": int(rec.num_points3D()),
               "camera": cams[0].model_name if cams else None,
               "size": [int(cams[0].width), int(cams[0].height)] if cams else None}
    return Report(all(p for _, p, _ in checks), checks, summary)
