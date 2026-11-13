"""Metric scale from Kinect depth (FR-6): metres per SfM unit, from SfM depth vs measured depth."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pycolmap

MIN_SAMPLES = 2000
MAX_MAD_OVER_S = 0.15
MAX_SESSION_DISAGREEMENT = 1.10
DEPTH_RANGE_MM = (500, 4000)


@dataclass
class ScaleResult:
    ok: bool
    scale: float | None = None
    n_samples: int = 0
    mad_over_s: float | None = None
    per_session: dict = field(default_factory=dict)
    residual_m: float | None = None
    reason: str = ""

    def as_dict(self) -> dict:
        return {"applied": self.ok, "scale_to_meters": self.scale, "samples": self.n_samples,
                "mad_over_scale": self.mad_over_s, "per_session": self.per_session,
                "median_abs_depth_error_m_after": self.residual_m, "reason": self.reason}


def collect_samples(rec: pycolmap.Reconstruction, depth_provider, session_of, min_track: int = 3):
    """-> list of (session, depth_m, z_sfm) over registered images with depth."""
    out = []
    for im in rec.images.values():
        if not im.has_pose:
            continue
        depth = depth_provider(im.name)
        if depth is None:
            continue
        T = im.cam_from_world().matrix()                     # 3x4
        h, w = depth.shape
        for p in im.points2D:
            if not p.has_point3D():
                continue
            pt = rec.points3D[p.point3D_id]
            if pt.track.length() < min_track:
                continue
            z = float(T[2, :3] @ pt.xyz + T[2, 3])
            if z <= 0:
                continue
            x, y = int(p.xy[0]), int(p.xy[1])
            if not (2 <= x < w - 2 and 2 <= y < h - 2):
                continue
            win = depth[y - 2:y + 3, x - 2:x + 3]
            valid = win[win > 0]
            if valid.size < 5:
                continue
            d = float(np.median(valid))
            if not (DEPTH_RANGE_MM[0] <= d <= DEPTH_RANGE_MM[1]):
                continue
            out.append((session_of(im.name), d / 1000.0, z))
    return out


def estimate_from_samples(samples, min_samples: int = MIN_SAMPLES) -> ScaleResult:
    if len(samples) < min_samples:
        return ScaleResult(False, n_samples=len(samples), reason=f"only {len(samples)} depth samples (need {min_samples})")
    sess = np.array([s[0] for s in samples])
    ratio = np.array([s[1] / s[2] for s in samples])
    z = np.array([s[2] for s in samples])
    depth = np.array([s[1] for s in samples])
    s = float(np.median(ratio))
    mad = float(np.median(np.abs(ratio - s)))
    per = {name: float(np.median(ratio[sess == name])) for name in sorted(set(sess))}
    res = ScaleResult(False, s, len(samples), mad / s, per)
    if mad / s > MAX_MAD_OVER_S:
        res.reason = f"depth and SfM disagree too much (MAD/scale {mad / s:.2f})"
        return res
    if len(per) > 1 and max(per.values()) / min(per.values()) > MAX_SESSION_DISAGREEMENT:
        res.reason = "sessions give different scales"
        return res
    res.ok = True
    res.residual_m = float(np.median(np.abs(depth - s * z)))
    return res


def apply_scale(rec: pycolmap.Reconstruction, s: float) -> None:
    rec.transform(pycolmap.Sim3d(float(s), pycolmap.Rotation3d(), np.zeros(3)))
