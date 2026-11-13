# kinect3dgs

Scan with an Xbox 360 Kinect (RGB-D) and produce the COLMAP dataset that 3D Gaussian Splatting tools train from: OpenSplat, gsplat, Nerfstudio, Postshot. This repo contains no splatting code; it ends at the dataset folder.

```
Kinect  ->  record (sessions: RGB + depth)  ->  dataset build (COLMAP)  ->  your 3DGS tool
```

## Why a depth camera helps

A phone video gives colour only. The Kinect also gives a depth map for every frame, registered to the colour image.

- **Real-world scale.** Photo-only SfM has arbitrary units. Here the depth fixes the scale, so the dataset (and the splat) is in metres. Measured on a table scan: three separate passes gave the same scale within 0.2 %, with a median depth error of 4.5 mm.
- **Depth is kept per frame.** Every recorded frame keeps its depth map, aligned to the colour image, for later steps (depth-seeded points, depth-aided poses). Today the depth is used for scale only.
- **Geometry without texture.** Depth does not need surface texture, which is where photo-only methods struggle (plain walls).

Limits: the colour image is 640x480 and soft, depth gets noisy beyond ~4 m (best 0.8-3.5 m), and the colour and depth sensors are not synchronised (up to half a frame apart; the skew is stored per frame). The Kinect needs its 12 V AC adapter and a USB 2.0 port.

## Setup

Needs `uv`, `cmake`, `libusb`, a C compiler and an Xbox 360 Kinect (tested: model 1473).

```bash
uv sync --inexact                      # Python 3.12 venv
bash scripts/build_libfreenect.sh      # libfreenect v0.7.5 into .local/ (no sudo, patched camera-only)
bash scripts/build_pywrapper.sh        # `freenect` Python module
sudo bash scripts/system_setup.sh      # udev rule + gspca_kinect blacklist; then replug the Kinect
bash scripts/usb_list.sh               # expect 045e:02ae (camera: FOUND)
uv run python -m kinect3dgs.probe      # grabs one RGB + depth frame and checks them
```

The first camera open takes 8-10 s (normal on the 1473). `freenect-camtest` does not work on this unit; use `probe`.

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

It picks the sharpest frames, runs COLMAP (pycolmap, CPU), applies the metric scale from the depth, undistorts, and writes `images/`, `sparse/0/` and `dataset.json`. About 4 minutes for ~180 frames.

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

## Status

Capture and dataset building are done. A minimal guided scan UI (live coaching, then one-click dataset build) is in progress. Design notes live in `specs/` (git-ignored). `data/` holds recordings and datasets (photos of your rooms) and is git-ignored.
