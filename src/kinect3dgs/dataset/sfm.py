"""pycolmap wrapper: features, matching tiers, mapping with retry (FR-3)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pycolmap

from .errors import QualityError
from .progress import Emitter

EXHAUSTIVE_MAX = 300
SEQ_OVERLAP = 20
LOOP_STRIDE = 4


@dataclass
class SfmResult:
    rec: pycolmap.Reconstruction
    mapper: str
    matching: str
    n_images: int
    registered: int
    n_points: int
    mean_error: float
    camera: dict
    dropped: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    attempts: dict = field(default_factory=dict)


def camera_info(cam: pycolmap.Camera) -> dict:
    return {"model": cam.model_name, "width": int(cam.width), "height": int(cam.height),
            "params_info": cam.params_info, "params": [round(float(p), 6) for p in cam.params]}


def pair_list(names: list[str], overlap: int = SEQ_OVERLAP, stride: int = LOOP_STRIDE) -> list[tuple[str, str]]:
    """Sequential pairs inside each session (names are '<session>_<idx>.png', sorted) + exhaustive over a stride subset."""
    by_session: dict[str, list[str]] = {}
    for n in names:
        by_session.setdefault(n.rsplit("_", 1)[0], []).append(n)
    pairs: set[tuple[str, str]] = set()
    for lst in by_session.values():
        for i, a in enumerate(lst):
            for b in lst[i + 1:i + 1 + overlap]:
                pairs.add((a, b))
    sub = names[::stride]
    for i, a in enumerate(sub):
        for b in sub[i + 1:]:
            pairs.add((a, b))
    return sorted(pairs)


def drop_outlier_cameras(rec: pycolmap.Reconstruction, factor: float = 10.0) -> list[str]:
    ids = list(rec.reg_image_ids())
    if len(ids) < 5:
        return []
    centers = np.array([rec.images[i].projection_center() for i in ids])
    d = np.linalg.norm(centers - np.median(centers, 0), axis=1)
    limit = factor * max(np.median(d), 1e-9)
    removed = []
    for i, dist in zip(ids, d):
        if dist > limit:
            removed.append(rec.images[i].name)
            rec.deregister_frame(rec.images[i].frame_id)
    return sorted(removed)


def _better(a, b):
    if a is None:
        return b
    if b is None:
        return a
    ka = (a.num_reg_images(), -a.compute_mean_reprojection_error())
    kb = (b.num_reg_images(), -b.compute_mean_reprojection_error())
    return b if kb > ka else a


def run_sfm(images_dir: Path, work: Path, max_image_size: int, matching: str, emit: Emitter,
            min_registered_frac: float = 0.8) -> SfmResult:
    work.mkdir(parents=True, exist_ok=True)
    pycolmap.logging.minloglevel = 2
    db = work / "database.db"
    db.unlink(missing_ok=True)
    names = sorted(p.name for p in images_dir.glob("*.png"))
    if len(names) < 8:
        raise QualityError(f"only {len(names)} frames selected; need at least 8 for SfM")
    timings: dict[str, float] = {}

    t = time.time()
    with emit.stage("features", f"{len(names)} images"):
        reader = pycolmap.ImageReaderOptions()
        reader.camera_model = "SIMPLE_RADIAL"
        ext = pycolmap.FeatureExtractionOptions()
        ext.max_image_size = max_image_size
        ext.sift.max_num_features = 8000
        pycolmap.extract_features(db, images_dir, image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
                                  reader_options=reader, extraction_options=ext)
    timings["features_s"] = round(time.time() - t, 1)

    t = time.time()
    mode = matching if matching != "auto" else ("exhaustive" if len(names) <= EXHAUSTIVE_MAX else "sequential")
    with emit.stage("matching", mode):
        emit.check()
        if mode == "exhaustive":
            pycolmap.match_exhaustive(db)
            used = "exhaustive"
        else:
            pairs = pair_list(names)
            pf = work / "pairs.txt"
            pf.write_text("".join(f"{a} {b}\n" for a, b in pairs))
            pycolmap.match_image_pairs(db, pairing_options=pycolmap.ImportedPairingOptions(match_list_path=str(pf)))
            used = f"sequential+loops ({len(pairs)} pairs)"
    timings["matching_s"] = round(time.time() - t, 1)

    t = time.time()
    best, kind, attempts = None, None, {}
    with emit.stage("mapping"):
        for k in ("incremental", "global"):
            emit.check()
            out = work / f"sparse_{k}"
            if out.exists():
                import shutil
                shutil.rmtree(out)
            out.mkdir(parents=True)
            models = (pycolmap.incremental_mapping(db, images_dir, out) if k == "incremental"
                      else pycolmap.global_mapping(db, images_dir, out))
            rec = max(models.values(), key=lambda r: r.num_reg_images()) if models else None
            attempts[k] = rec.num_reg_images() if rec else 0
            if rec is not None and _better(best, rec) is rec:
                best, kind = rec, k
            if best is not None and best.num_reg_images() >= min_registered_frac * len(names):
                break
    timings["mapping_s"] = round(time.time() - t, 1)
    if best is None:
        raise QualityError("SfM produced no model: not enough overlap between frames")

    dropped = drop_outlier_cameras(best)
    if dropped:
        try:
            pycolmap.bundle_adjustment(best)
        except Exception:
            pass
    cam = next(iter(best.cameras.values()))
    return SfmResult(best, kind, used, len(names), best.num_reg_images(), best.num_points3D(),
                     float(best.compute_mean_reprojection_error()), camera_info(cam), dropped, timings, attempts)
