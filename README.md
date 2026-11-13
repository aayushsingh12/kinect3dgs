# kinect3dgs

Scan with an Xbox 360 Kinect (RGB-D) and produce the COLMAP dataset that 3D Gaussian Splatting tools train from: OpenSplat, gsplat, Nerfstudio, Postshot. The Kinect adds a depth map to every frame, which gives the dataset real-world scale (metres, not arbitrary units) and is kept for later depth-aided steps; today depth is used for scale only. This repo contains no splatting code; it ends at the dataset folder.

```
Kinect  ->  record (sessions: RGB + depth)  ->  dataset build (COLMAP)  ->  your 3DGS tool
```

## Setup

Needs `uv`, `cmake`, `libusb`, a C compiler and an Xbox 360 Kinect (tested: model 1473) with its 12 V AC adapter on a USB 2.0 port. Colour is 640x480; depth is best at 0.8-3.5 m and noisy beyond ~4 m.

```bash
uv sync --inexact                      # Python 3.12 venv
bash scripts/build_libfreenect.sh      # libfreenect v0.7.5 into .local/ (no sudo, patched camera-only)
bash scripts/build_pywrapper.sh        # `freenect` Python module
sudo bash scripts/system_setup.sh      # udev rule + gspca_kinect blacklist; then replug the Kinect
bash scripts/usb_list.sh               # expect 045e:02ae (camera: FOUND)
uv run python -m kinect3dgs.probe      # grabs one RGB + depth frame and checks them
```

The first camera open takes 8-10 s (normal on the 1473). `freenect-camtest` does not work on this unit; use `probe`.

### Tested configuration

| | |
|---|---|
| Sensor | Xbox 360 Kinect, model **1473**; on USB it shows a hub (`045e:02c2`), audio (`045e:02ad`) and camera (`045e:02ae`), no separate motor device |
| Connection | plugged straight into the laptop, USB 2.0 (480 Mb/s), with the 12 V AC adapter |
| Streams | RGB 640x480 and registered depth 640x480 (mm), both at a steady 30 fps |
| Driver | libfreenect v0.7.5 with local patches: camera only (no motor, LED, tilt or audio) and a `set_flag` wrapper |
| Host | Arch Linux, kernel 7.2, Python 3.12, pycolmap 4.2.1 (CPU); OpenSplat trained on an RTX 3050 Laptop GPU (4 GB) |

Not tested: the 1414 model, Kinect for Windows, other operating systems. Exposure lock has no effect on the 1473, so auto exposure is used.

## Guided scan UI

The easiest way to use all of this: a small local page (dark, minimal, keyboard-driven, `127.0.0.1` only) that guides the scan, then builds the dataset.

```bash
uv run python -m kinect3dgs.ui --open            # real Kinect
uv run python -m kinect3dgs.ui --demo --open     # synthetic camera, no hardware
```

Flow: check -> setup (name, table/room, output folder; default `data/datasets/<name>`) -> guided passes -> build -> copy the OpenSplat command. Keys: `space` record/pause, `n` next pass, `f` finish, `d` depth view, `p` build, `esc` cancel. Each pass is its own session (`<name>_p1..p3`). One live hint at a time: too close, too far, slow down, blurry, too dark, too bright, no depth.

The sections below are the same steps by command line.

## 1. Record a scan

```bash
uv run python -m kinect3dgs.record --seconds 90 --preview --name table01
```

- Records RGB + registered depth (16-bit PNG, millimetres, 0 = invalid) at ~10 fps (`--every 3` of 30 fps) into `data/sessions/<name>/`.
- `--preview` shows a live RGB/depth view at `http://127.0.0.1:8765/` (loopback only).
- `--mode snapshot`: press Enter to save one pair (`q` to finish).
- Ctrl-C stops cleanly. If the cable is knocked out, it waits up to 90 s for you to plug it back in and carries on.
- Every frame stores a sharpness score and the RGB/depth time skew.

Scan tips: move slowly and smoothly, keep 0.8-1.5 m from small subjects (1-3.5 m for rooms), walk around rather than turning on the spot, use steady light, and do two or three passes at different heights. Several sessions can go into one dataset.

## 2. Build the COLMAP dataset

```bash
uv run python -m kinect3dgs.dataset build --sessions table01 table02 table03 --out ~/datasets/table
uv run python -m kinect3dgs.dataset validate ~/datasets/table
uv run python -m kinect3dgs.dataset info ~/datasets/table
```

It picks the sharpest frames, runs COLMAP (pycolmap, CPU), applies the metric scale from the depth, undistorts, and writes `images/`, `sparse/0/` and `dataset.json`. About 1.5-4 minutes for ~200 frames.

- `--out` chooses the folder (default `data/datasets/<name>`). A non-empty folder is refused unless it is an old dataset and you pass `--force`; it never writes into the project's source folders. A failed or cancelled build leaves `--out` untouched.
- Quality gates fail the build with advice when fewer than 80 % of frames register, there are too few points, or the error is high.
- Options: `--target-frames`, `--no-metric-scale`, `--text` (also write COLMAP `.txt`), `--progress-jsonl` (machine-readable progress; SIGTERM cancels).

## 3. Train

```bash
opensplat ~/datasets/table -n 30000 -o splat.ply
```

Verified with OpenSplat. The other tools read COLMAP datasets too but have not been tested with this output.

## Other tools

- `python -m kinect3dgs.exposure_check` tests whether the camera's exposure lock works (it does not on the 1473, so auto exposure is the default).
- `scripts/clean_shm.sh` removes leaked shared memory after a hard kill.
- Tests: `uv run pytest -m "not hardware and not colmap"` (fast), `-m colmap` (synthetic end-to-end, ~70 s), `KINECT_HW=1 uv run pytest -m hardware` (needs a Kinect).

## Results (measured on the author's Kinect)

| Scene | Frames | Registered | Points | Error | Metric scale | OpenSplat (30 000 it.) |
|---|---|---|---|---|---|---|
| table, 3 passes of 90 s | 179 | 167 (93 %) | 17 235 | 0.71 px | 70 k depth samples, 4.5 mm median error | 537 s, 44.5 k Gaussians |
| bedroom, 3 passes of ~3.5 min (guided UI) | 197 | 197 (100 %) | 17 749 | 0.79 px | 120 k samples, passes agree within 0.2 %, 2.5 cm median error | 683 s, 54.1 k Gaussians |

Dataset builds took 231 s (table) and about 100 s (bedroom) on the CPU; training used an RTX 3050 Laptop GPU. Splats look like the photos where the cameras looked (objects and furniture sharp; fabric texture and far background softer, a little haze where few cameras looked). The colour camera is 640x480 and noisy, so splats are only as sharp as that input.

## Limits and ideas

- Depth is used for metric scale only. Seeding the point cloud from depth and depth-aided poses are the next quality steps.
- Only OpenSplat is verified with the generated folders; Nerfstudio, gsplat and Postshot should read them but are untested.
- Matching is exhaustive up to 300 frames; above that a sequential + loop-closure mode is used that has not been tested on real data.
- The 1473's colour sensor ignores the exposure lock, the first camera open takes 8-10 s, and RGB and depth frames are up to half a frame (~16 ms) apart (stored per frame as `skew_ms`).
- Not tested on hardware: the "plug the cable back in" banner in the UI, and how well the coaching hints match real conditions beyond the author's own scans.

## Troubleshooting

| Symptom | What to do |
|---|---|
| `camera: MISSING` in `usb_list.sh` | the Kinect needs its 12 V adapter plugged in at the wall and at the Kinect; replug the USB cable |
| `Can't open device` / `LIBUSB_ERROR_ACCESS` | run `sudo bash scripts/system_setup.sh`, then replug |
| Build fails: "only N/M frames registered" | scan slower with more overlap, keep 0.8-1.5 m from small subjects, steady light |
| Splat hazy in places | add passes covering those angles; keep frames sharp |
| UI page says forbidden (403) | reload the page (the token changes every time the server starts) |
| Leftover `/dev/shm/kinect3dgs-*` | `bash scripts/clean_shm.sh` |

## Project layout

```
src/kinect3dgs/   sensor, stream (Kinect I/O), recorder + record (sessions), probe,
                  dataset/ (COLMAP dataset builder), ui/ (guided scan page)
scripts/          libfreenect build, system setup, USB listing
patches/          local libfreenect patches (camera only, set_flag)
tools/            offline helpers (coaching threshold tuning)
tests/            hardware-free tests (fake camera, synthetic scene); hardware tests are opt-in
data/             sessions, datasets, work files: git-ignored, contains photos of your rooms
```

Design notes live in `specs/` (git-ignored, written per feature: handoff, PRD, plan, pseudocode, results).

