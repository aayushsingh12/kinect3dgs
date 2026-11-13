"""Derive the coaching speed threshold from recorded sessions: python tools/tune_coach.py table01 table02 table03

For consecutive recorded frames (3 camera frames apart at the default --every 3) it measures the image shift with
phase correlation and compares it with a blur indicator (sharpness / rolling median). Prints shift bins with the
fraction of blurry frames and the shift where that fraction crosses 50 %.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

GAP = 3                       # camera frames between recorded frames
WIN = 40                      # recorded frames = 4 s at 10 fps


def session_arrays(root: Path):
    rows = [json.loads(l) for l in (root / "frames.jsonl").read_text().splitlines()]
    sharp = np.array([r["sharpness"] for r in rows])
    prev, shifts = None, []
    for r in rows:
        g = cv2.cvtColor(cv2.imread(str(root / "rgb" / f"{r['idx']:06d}.png")), cv2.COLOR_BGR2GRAY)
        s = cv2.resize(g, (160, 120)).astype(np.float32)
        shifts.append(0.0 if prev is None else float(np.hypot(*cv2.phaseCorrelate(prev, s)[0]) * 4.0 / GAP))
        prev = s
    ratio = np.array([sharp[i] / np.median(sharp[max(0, i - WIN):i + 1]) for i in range(len(sharp))])
    return np.array(shifts)[1:], ratio[1:], sharp[1:]


def main():
    names = sys.argv[1:] or ["table01", "table02", "table03"]
    S, R = [], []
    for n in names:
        s, r, _ = session_arrays(Path("data/sessions") / n)
        S.append(s); R.append(r)
        print(f"{n}: {len(s)} frame pairs, shift px/frame median {np.median(s):.1f}  p90 {np.percentile(s, 90):.1f}")
    S, R = np.concatenate(S), np.concatenate(R)
    blurry = R < 0.5
    print(f"\nall: {len(S)} pairs, blurry (ratio < 0.5) {blurry.mean():.0%}, correlation(shift, ratio) {np.corrcoef(S, R)[0, 1]:.2f}")
    edges = [0, 2, 4, 6, 8, 10, 12, 14, 17, 20, 25, 40, 1e9]
    for a, b in zip(edges[:-1], edges[1:]):
        m = (S >= a) & (S < b)
        if m.sum():
            print(f"  shift {a:5.1f}-{min(b, 999):5.1f} px/frame  n={m.sum():4d}  blurry {blurry[m].mean():4.0%}  median ratio {np.median(R[m]):.2f}")
    # smallest shift above which >= 50 % of frames are blurry (using a sliding window of 40 sorted samples)
    order = np.argsort(S)
    Ss, Bs = S[order], blurry[order]
    cross = None
    for i in range(0, len(Ss) - 40):
        if Bs[i:i + 40].mean() >= 0.5:
            cross = float(Ss[i]); break
    print(f"\nshift where >= 50 % of frames are blurry: {cross if cross is not None else 'not reached'} px/frame")


if __name__ == "__main__":
    main()
