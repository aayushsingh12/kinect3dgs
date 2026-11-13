"""python -m kinect3dgs.dataset build|validate|info"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import build_dataset, default_params
from .errors import Cancelled, DatasetError, UsageError
from .progress import Emitter
from .validate import validate_dataset


def parser():
    ap = argparse.ArgumentParser(prog="kinect3dgs.dataset", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="sessions -> COLMAP dataset")
    b.add_argument("--sessions", nargs="+", required=True, help="session names (under data/sessions) or paths")
    b.add_argument("--out", help="output folder (default data/datasets/<name>)")
    b.add_argument("--name")
    b.add_argument("--work-dir")
    b.add_argument("--target-frames", type=int, default=200)
    b.add_argument("--min-sharpness-frac", type=float, default=0.6)
    b.add_argument("--max-image-size", type=int, default=800)
    b.add_argument("--matching", choices=["auto", "exhaustive", "sequential"], default="auto")
    b.add_argument("--no-metric-scale", action="store_true")
    b.add_argument("--text", action="store_true", help="also write cameras/images/points3D .txt")
    b.add_argument("--force", action="store_true", help="replace an existing dataset in --out")
    b.add_argument("--clean-work", action="store_true")
    b.add_argument("--progress-jsonl", action="store_true")
    b.add_argument("--quiet", action="store_true")
    v = sub.add_parser("validate", help="check a dataset folder")
    v.add_argument("path")
    v.add_argument("--json", action="store_true")
    i = sub.add_parser("info", help="summarise a dataset folder")
    i.add_argument("path")
    return ap


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    if a.cmd == "validate":
        rep = validate_dataset(a.path)
        if a.json:
            print(json.dumps(rep.as_dict(), indent=2))
        else:
            for n, p, d in rep.checks:
                print(f"{'ok  ' if p else 'FAIL'} {n} {d}")
            print("VALID" if rep.ok else "INVALID")
        return 0 if rep.ok else 1
    if a.cmd == "info":
        p = Path(a.path) / "dataset.json"
        if not p.exists():
            print(f"no dataset.json in {a.path}")
            return 2
        m = json.loads(p.read_text())
        s = m["sfm"]
        print(f"{m['name']}  created {m['created_utc']}\n  sessions: {', '.join(m['sessions'])}\n"
              f"  frames: {s['registered']}/{s['selected']} registered, {s['points']} points, error {s['mean_error_px']} px\n"
              f"  camera: {m['camera']['model']} {m['camera']['width']}x{m['camera']['height']} f~{m['camera']['params'][0]:.1f}\n"
              f"  metric scale: {m['metric_scale'].get('scale_to_meters') if m['metric_scale'].get('applied') else 'not applied'}")
        for w in m.get("warnings", []):
            print("  warning:", w)
        return 0
    emit = Emitter(jsonl=a.progress_jsonl, quiet=a.quiet)
    emit.install_signals()
    try:
        params = default_params(a.sessions, a.out, a.name, work_dir=a.work_dir, target_frames=a.target_frames,
                                min_sharpness_frac=a.min_sharpness_frac, max_image_size=a.max_image_size,
                                matching=a.matching, metric_scale=not a.no_metric_scale, text=a.text, force=a.force,
                                clean_work=a.clean_work, quiet=a.quiet, progress_jsonl=a.progress_jsonl)
        r = build_dataset(params, emitter=emit)
        if not a.progress_jsonl and not a.quiet:
            print(f"\nDataset written to {r['out']}\n  {r['registered']}/{r['selected']} frames registered, {r['points']} points, "
                  f"error {r['mean_error_px']} px, metric scale {'yes' if r['metric_scale'] else 'no'}, {r['elapsed_s']} s")
            for w in r["warnings"]:
                print("  warning:", w)
            print(f"\nTrain with OpenSplat:\n  opensplat {r['out']} -n 30000 -o splat.ply")
        return 0
    except Cancelled:
        emit.result({"ok": False, "error": "cancelled", "exit_code": 130})
        print("cancelled", file=sys.stderr)
        return 130
    except DatasetError as e:
        emit.result({"ok": False, "error": str(e), "exit_code": e.exit_code})
        print("FAIL:", e, file=sys.stderr)
        return e.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
