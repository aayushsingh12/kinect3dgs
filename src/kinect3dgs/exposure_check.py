"""Interactive check that the exposure/white-balance lock works: python -m kinect3dgs.exposure_check

For each mode (lock, auto) the script streams, then asks you to cover about HALF of the middle (colour) lens
for a few seconds, either side. With auto exposure the camera brightens the uncovered half to compensate;
with the lock it must not. Reports the brightness change of the uncovered half in each mode.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from .stream import KinectStream

BASELINE_S, COVER_S, RECOVER_S = 8.0, 8.0, 5.0


def run_mode(lock: bool) -> dict:
    ks = KinectStream(lock_exposure=lock, lock_after_s=3.0)
    series = []
    stage = 0
    with ks:
        t0 = None
        while True:
            p = ks.get_pair(2.0)
            if p is None:
                continue
            t0 = t0 or time.monotonic()
            t = time.monotonic() - t0
            g = p.rgb.mean(axis=2)
            series.append((t, float(g[:, 384:].mean()), float(g[:, :256].mean())))   # t, right, left
            if stage == 0 and t >= BASELINE_S:
                print("\n>>> COVER about HALF of the MIDDLE (colour) lens now, with the edge of a hand or card, either side <<<", flush=True)
                stage = 1
            elif stage == 1 and t >= BASELINE_S + COVER_S:
                print(">>> UNCOVER <<<", flush=True)
                stage = 2
            elif stage == 2 and t >= BASELINE_S + COVER_S + RECOVER_S:
                break
    a = np.array(series)
    w0 = (a[:, 0] >= BASELINE_S - 2) & (a[:, 0] < BASELINE_S)
    w1 = (a[:, 0] >= BASELINE_S + 3.5) & (a[:, 0] < BASELINE_S + COVER_S)
    change = {side: float((a[w1, col].mean() - a[w0, col].mean()) / a[w0, col].mean() * 100)
              for side, col in (("right", 1), ("left", 2))}
    covered = min(change, key=change.get)                 # the half that got darker
    other = "right" if covered == "left" else "left"
    return {"mode": "lock" if lock else "auto", "covered_side": covered, "covered_change_pct": change[covered],
            "uncovered_change_pct": change[other], "lock_results": ks.stats.lock_results}


def main() -> int:
    only = sys.argv[1] if len(sys.argv) > 1 else "both"       # both | lock | auto
    print("Exposure check: two runs (lock, auto), ~35 s each. Point the Kinect at a normal lit scene and keep it still.")
    res = {}
    for lock in {"both": (True, False), "lock": (True,), "auto": (False,)}[only]:
        r = run_mode(lock)
        res[r["mode"]] = r
        print(json.dumps(r))
    if only != "both":
        r = res[only]
        print(f"\n{only}: covered half {r['covered_change_pct']:+.1f}%, uncovered half {r['uncovered_change_pct']:+.1f}%")
        return 0
    dl, da = abs(res["lock"]["uncovered_change_pct"]), abs(res["auto"]["uncovered_change_pct"])
    if res["lock"]["covered_change_pct"] > -15 or res["auto"]["covered_change_pct"] > -15:
        verdict = "INCONCLUSIVE: the covered half did not get dark enough; cover more of the lens"
    elif da < 3:
        verdict = "INCONCLUSIVE: auto exposure did not react, so the lock cannot be distinguished"
    elif dl < 3 and da - dl > 3:
        verdict = "LOCK EFFECTIVE: locked brightness stayed put while auto exposure compensated"
    else:
        verdict = "LOCK NOT EFFECTIVE: brightness changed with the lock on"
    print(f"\nuncovered-half change: lock {res['lock']['uncovered_change_pct']:+.1f}%  auto {res['auto']['uncovered_change_pct']:+.1f}%")
    print("VERDICT:", verdict)
    return 0 if verdict.startswith("LOCK EFFECTIVE") else 1


if __name__ == "__main__":
    sys.exit(main())
