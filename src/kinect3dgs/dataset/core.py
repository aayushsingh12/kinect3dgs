"""build_dataset: sessions -> COLMAP dataset (library API, FR-12)."""
from __future__ import annotations

import shutil
import os
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from . import export, scale as scale_mod, sfm as sfm_mod
from .errors import Cancelled, QualityError, UsageError
from .params import REPO, BuildParams, check_paths, resolve_session
from .progress import Emitter
from .select import load_rows, materialise, select_frames
from .validate import validate_dataset

MIN_REGISTERED_FRAC = 0.8
WARN_REGISTERED_FRAC = 0.9
MIN_POINTS = 3000
FAIL_ERROR_PX, WARN_ERROR_PX = 2.0, 1.0


def default_params(sessions: list[str], out: str | None = None, name: str | None = None, **kw) -> BuildParams:
    dirs = [resolve_session(s) for s in sessions]
    name = name or "_".join(d.name for d in dirs)[:60]
    outp = Path(out) if out else REPO / "data" / "datasets" / name
    work = Path(kw.pop("work_dir", None) or REPO / "data" / "work" / name)
    if not out:                                   # default locations may create their own parents (data/ is git-ignored)
        outp.parent.mkdir(parents=True, exist_ok=True)
    work.parent.mkdir(parents=True, exist_ok=True)
    return BuildParams(sessions=dirs, out=outp, name=name, work_dir=work, **kw)


def evaluate_gates(res: sfm_mod.SfmResult, per_session_selected: dict, per_session_registered: dict):
    """-> (warnings, verdicts). Raises QualityError when a hard gate fails."""
    warnings, fails = [], []
    frac = res.registered / max(1, res.n_images)
    if frac < MIN_REGISTERED_FRAC:
        fails.append(f"only {res.registered}/{res.n_images} frames registered ({frac:.0%}, need {MIN_REGISTERED_FRAC:.0%}): "
                     "scan slower with more overlap, keep 0.8-1.5 m from the subject, use steady light")
    if res.n_points < MIN_POINTS:
        fails.append(f"only {res.n_points} 3D points (need {MIN_POINTS}): the scene may lack texture")
    if res.mean_error > FAIL_ERROR_PX:
        fails.append(f"mean reprojection error {res.mean_error:.2f} px is above {FAIL_ERROR_PX}: blurry frames or motion")
    if frac >= MIN_REGISTERED_FRAC and frac < WARN_REGISTERED_FRAC:
        warnings.append(f"{frac:.0%} of frames registered (below {WARN_REGISTERED_FRAC:.0%})")
    if WARN_ERROR_PX < res.mean_error <= FAIL_ERROR_PX:
        warnings.append(f"mean reprojection error {res.mean_error:.2f} px is high")
    for s, n in per_session_selected.items():
        r = per_session_registered.get(s, 0)
        if n and r / n < 0.5:
            warnings.append(f"session {s}: only {r}/{n} frames registered (poorly connected; light changed between passes?)")
    verdicts = {"registered_fraction": round(frac, 4), "points": res.n_points, "mean_error_px": round(res.mean_error, 4),
                "passed": not fails}
    if fails:
        raise QualityError("; ".join(fails))
    return warnings, verdicts


def _depth_provider(session_dirs: dict[str, Path]):
    def get(image_name: str):
        stem = image_name.rsplit(".", 1)[0]
        session, idx = stem.rsplit("_", 1)
        f = session_dirs[session] / "depth" / f"{int(idx):06d}.png"
        return cv2.imread(str(f), cv2.IMREAD_UNCHANGED) if f.exists() else None
    return get


def build_dataset(params: BuildParams, on_progress=None, emitter: Emitter | None = None) -> dict:
    emit = emitter or Emitter(callback=on_progress, jsonl=params.progress_jsonl, quiet=params.quiet)
    check_paths(params)
    session_dirs = {Path(s).name: Path(s) for s in params.sessions}
    work, out = params.work_dir, params.out
    tmp_out = out.with_name(out.name + f".tmp-{os.getpid()}")
    try:
        # ---- select
        with emit.stage("select", f"{len(session_dirs)} session(s)"):
            rows = {n: load_rows(d) for n, d in session_dirs.items()}
            sel = select_frames(rows, params.target_frames, params.min_sharpness_frac)
            if len(sel.picks) < 8:
                raise QualityError(f"only {len(sel.picks)} usable frames selected; record more or lower --min-sharpness-frac")
            images = work / "images"
            if images.exists():
                shutil.rmtree(images)
            materialise(sel, session_dirs, images)
            emit.event("select", "progress", 1.0, f"{len(sel.picks)} frames, {len(sel.skipped)} windows skipped")

        # ---- sfm
        res = sfm_mod.run_sfm(images, work, params.max_image_size, params.matching, emit, MIN_REGISTERED_FRAC)
        per_sel = {s: sum(1 for p in sel.picks if p.session == s) for s in session_dirs}
        reg_names = [im.name for im in res.rec.images.values() if im.has_pose]
        per_reg = {s: sum(1 for n in reg_names if n.rsplit("_", 1)[0] == s) for s in session_dirs}
        warnings, verdicts = evaluate_gates(res, per_sel, per_reg)
        warnings = list(sel.warnings) + warnings

        # ---- metric scale
        with emit.stage("scale", "metric scale from Kinect depth" if params.metric_scale else "disabled"):
            if params.metric_scale:
                samples = scale_mod.collect_samples(res.rec, _depth_provider(session_dirs), lambda n: n.rsplit("_", 1)[0])
                sres = scale_mod.estimate_from_samples(samples)
                if sres.ok:
                    scale_mod.apply_scale(res.rec, sres.scale)
                else:
                    warnings.append(f"metric scale not applied: {sres.reason}")
                scale_meta = sres.as_dict()
            else:
                scale_meta = {"applied": False, "reason": "disabled by --no-metric-scale"}

        # ---- undistort + export
        emit.check()
        with emit.stage("undistort"):
            export.export_dataset(res.rec, images, work, tmp_out, params.text)
        with emit.stage("export"):
            final = pycolmap.Reconstruction(tmp_out / "sparse" / "0")
            meta = {
                "schema_version": 1, "name": params.name, "created_utc": export.now_utc(),
                "versions": {"pycolmap": pycolmap.__version__, "numpy": np.__version__},
                "sessions": {n: str(d) for n, d in session_dirs.items()},
                "selection": {"picks": [vars(p) for p in sel.picks], "skipped_windows": len(sel.skipped),
                              "per_session": sel.per_session},
                "params": {k: getattr(params, k) for k in ("target_frames", "min_sharpness_frac", "max_image_size",
                                                           "matching", "metric_scale")},
                "camera_before_undistortion": res.camera,
                "camera": sfm_mod.camera_info(next(iter(final.cameras.values()))),
                "sfm": {"selected": res.n_images, "registered": final.num_reg_images(), "points": final.num_points3D(),
                        "mean_error_px": round(res.mean_error, 4), "mapper": res.mapper, "matching": res.matching,
                        "dropped_outliers": res.dropped, "timings": res.timings, "attempts": res.attempts,
                        "registered_per_session": per_reg},
                "metric_scale": scale_meta, "quality": verdicts, "warnings": warnings,
            }
            export.write_metadata(tmp_out, meta, out.name)
        with emit.stage("validate"):
            rep = validate_dataset(tmp_out)
            if not rep.ok:
                bad = "; ".join(f"{n}: {d}" for n, p, d in rep.checks if not p)
                raise QualityError(f"generated dataset failed validation ({bad})")
        emit.check()
        export.commit(tmp_out, out)
        summary = {"ok": True, "out": str(out), "registered": meta["sfm"]["registered"], "selected": res.n_images,
                   "points": meta["sfm"]["points"], "mean_error_px": meta["sfm"]["mean_error_px"],
                   "metric_scale": scale_meta.get("applied", False), "warnings": warnings,
                   "elapsed_s": round(emit.events[-1]["elapsed_s"], 1)}
        emit.result(summary)
        if params.clean_work:
            shutil.rmtree(work, ignore_errors=True)
        return summary
    except Cancelled:
        raise
    finally:
        if tmp_out.exists():
            shutil.rmtree(tmp_out, ignore_errors=True)
