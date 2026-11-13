"""Deterministic synthetic RGB-D scene with known poses, for tests and demos (no real photos involved).

A textured ground plane plus two textured boxes, ray-cast from cameras on an arc. Depth maps are exact z-depth in mm.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

W, H, F = 640, 480, 540.0
K_RADIAL = 0.03


@dataclass
class Scene:
    images: list            # uint8 (H,W,3) RGB
    depth_mm: list          # uint16 (H,W), 0 = no surface
    c2w: list               # 4x4 camera-to-world, camera x right, y down, z forward (COLMAP convention)
    scale_m: float
    names: list


def _texture(seed: int, size: int = 1024) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), np.float32)
    for cells, amp in ((8, 0.5), (16, 0.45), (32, 0.4), (64, 0.35), (128, 0.3), (256, 0.25)):
        base = rng.random((cells, cells, 3)).astype(np.float32)
        img += amp * cv2.resize(base, (size, size), interpolation=cv2.INTER_CUBIC)
    for _ in range(120):                                   # distinctive blobs help SIFT
        c = tuple(int(v) for v in rng.integers(0, size, 2))
        cv2.circle(img, c, int(rng.integers(6, 40)), tuple(float(v) for v in rng.random(3) * 2), -1 if rng.random() < .5 else 3)
    img = img - img.min()
    return (255 * img / img.max()).astype(np.uint8)


def _lookat(cam: np.ndarray, target: np.ndarray) -> np.ndarray:
    f = target - cam
    f /= np.linalg.norm(f)
    right = np.cross(f, np.array([0, 0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(f, right)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = right, down, f, cam
    return T


def _undistort_norm(xd: np.ndarray, yd: np.ndarray, k: float):
    xu, yu = xd.copy(), yd.copy()
    for _ in range(8):
        r2 = xu * xu + yu * yu
        xu, yu = xd / (1 + k * r2), yd / (1 + k * r2)
    return xu, yu


def _scene_geometry(scale: float):
    a, b = 1.0 * scale, 0.75 * scale
    boxes = [(np.array([-0.55, -0.05, 0.0]) * scale, np.array([-0.2, 0.3, 0.3]) * scale),
             (np.array([0.2, -0.45, 0.0]) * scale, np.array([0.5, -0.15, 0.4]) * scale)]
    return a, b, boxes


def render_view(c2w: np.ndarray, scale: float, textures) -> tuple[np.ndarray, np.ndarray]:
    a, b, boxes = _scene_geometry(scale)
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    xd, yd = (u - W / 2) / F, (v - H / 2) / F
    xu, yu = _undistort_norm(xd, yd, K_RADIAL)
    dir_c = np.stack([xu, yu, np.ones_like(xu)], -1).reshape(-1, 3)      # z component 1 => t equals z-depth
    d = dir_c @ c2w[:3, :3].T
    o = c2w[:3, 3]
    n = d.shape[0]
    best_t = np.full(n, np.inf)
    colour = np.full((n, 3), 40.0)
    def sample(tex, uu, vv):          # remap needs maps shaped like an image (rows < 32767)
        mu = np.nan_to_num(uu, nan=0.0, posinf=0.0, neginf=0.0).reshape(H, W).astype(np.float32)
        mv = np.nan_to_num(vv, nan=0.0, posinf=0.0, neginf=0.0).reshape(H, W).astype(np.float32)
        return cv2.remap(tex, mu, mv, cv2.INTER_LINEAR).reshape(-1, 3)
    # ground plane z = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -o[2] / d[:, 2]
    p = o + t[:, None] * d
    ok = (t > 0) & (np.abs(p[:, 0]) <= a) & (np.abs(p[:, 1]) <= b)
    tex = textures[0]
    s = tex.shape[0] - 1
    c = sample(tex, (p[:, 0] + a) / (2 * a) * s, (p[:, 1] + b) / (2 * b) * s)
    best_t[ok] = t[ok]
    colour[ok] = c[ok]
    # boxes (slab method)
    for bi, (lo, hi) in enumerate(boxes):
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (lo - o) / d, (hi - o) / d
        tn, tf = np.minimum(t1, t2), np.maximum(t1, t2)
        tmin, tmax = tn.max(1), tf.min(1)
        ok = (tmax > tmin) & (tmin > 0) & (tmin < best_t)
        axis = tn.argmax(1)
        p = o + tmin[:, None] * d
        tex = textures[1 + bi]
        s = tex.shape[0] - 1
        uu = np.where(axis == 0, p[:, 1], p[:, 0])
        vv = np.where(axis == 2, p[:, 1], p[:, 2])
        span = (hi - lo).max()
        c = sample(tex, ((uu - uu.min()) % (span * 1.0) / span) * s, ((vv - vv.min()) % (span * 1.0) / span) * s)
        best_t[ok] = tmin[ok]
        colour[ok] = c[ok]
    hit = np.isfinite(best_t)
    depth = np.zeros(n, np.uint16)
    depth[hit] = np.clip(best_t[hit] * 1000.0, 1, 65535).astype(np.uint16)
    img = np.clip(colour, 0, 255).astype(np.uint8).reshape(H, W, 3)
    return img, depth.reshape(H, W)


def make_scene(n_views: int = 24, scale_m: float = 2.0, seed: int = 0) -> Scene:
    textures = [_texture(seed + i) for i in range(3)]
    rng = np.random.default_rng(seed)
    per_ring = n_views // 2
    c2w, images, depths, names = [], [], [], []
    for ring, elev in enumerate((38.0, 58.0)):
        for i in range(per_ring):
            az = np.deg2rad(-65 + 130 * i / max(1, per_ring - 1) + rng.uniform(-3, 3))
            el = np.deg2rad(elev + rng.uniform(-2, 2))
            r = 1.7 * scale_m * (1 + rng.uniform(-0.03, 0.03))
            cam = np.array([r * np.cos(el) * np.sin(az), -r * np.cos(el) * np.cos(az), r * np.sin(el)])
            T = _lookat(cam, np.array([0.0, -0.1 * scale_m, 0.12 * scale_m]))
            img, dep = render_view(T, scale_m, textures)
            c2w.append(T); images.append(img); depths.append(dep)
            names.append(f"v{ring}{i:02d}")
    return Scene(images, depths, c2w, scale_m, names)


def write_session(root: Path, scene: Scene) -> Path:
    """Write the scene as an AKS-002 session folder (schema_version 1)."""
    root = Path(root)
    (root / "rgb").mkdir(parents=True, exist_ok=True)
    (root / "depth").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (img, dep) in enumerate(zip(scene.images, scene.depth_mm)):
        cv2.imwrite(str(root / "rgb" / f"{i:06d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(root / "depth" / f"{i:06d}.png"), dep)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        rows.append({"idx": i, "t_host_s": i * 0.1, "ts_rgb": i, "ts_depth": i, "skew_ms": 0.0, "host_skew_ms": 0.0,
                     "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                     "depth_valid_frac": float((dep > 0).mean()), "bad_pair": False})
    (root / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "session.json").write_text(json.dumps({
        "schema_version": 1, "started_utc": "synthetic", "finished_utc": "synthetic", "aborted": None,
        "formats": {"rgb": {"mode": "VIDEO_RGB"}, "depth": {"mode": "DEPTH_REGISTERED", "units": "mm", "invalid": 0}},
        "args": {}, "stats": {}, "events": [], "intrinsics": None}))
    return root
