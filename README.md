# kinect3dgs
Using XBOX 360's Kinect camera (RGBD) to perform 3D Gaussian Splatting

## Getting started (AKS-001)

Needs: Xbox 360 Kinect (tested: model 1473) with its 12 V AC adapter, plugged directly into a USB 2.0 port; `uv`, `cmake`, `libusb`, a C compiler.

```bash
uv sync --inexact                      # venv on Python 3.12
bash scripts/build_libfreenect.sh      # libfreenect v0.7.5 -> .local/ (no sudo)
bash scripts/build_pywrapper.sh        # `freenect` Python module into the venv
sudo bash scripts/system_setup.sh      # udev rule + gspca_kinect blacklist; then replug the Kinect
bash scripts/system_setup.sh --check   # verify (no sudo)
bash scripts/usb_list.sh               # expect 045e:02ae (camera)
uv run python -m kinect3dgs.probe      # writes data/probe/<timestamp>/ (git-ignored)
uv run pytest                          # KINECT_HW=1 uv run pytest -m hardware  with a live Kinect
```

Notes: the first open of a 1473 unit takes ~6 s (libfreenect resets the audio sibling device); `freenect-camtest` is not usable on this unit because it also opens the audio device; motor/LED/tilt are disabled by `patches/0001-camera-only-subdevices.patch`. See `specs/` (git-ignored) for the full write-up.

## Recording a session (AKS-002)

After pulling patch 0002, rebuild once: `bash scripts/build_libfreenect.sh && bash scripts/build_pywrapper.sh`.

```bash
uv run python -m kinect3dgs.record --seconds 60 --name scan01            # ~10 fps, exposure locked, registered depth
uv run python -m kinect3dgs.record --seconds 60 --preview                # live view at http://127.0.0.1:8765/
uv run python -m kinect3dgs.record --mode snapshot --name calib01        # Enter = save a pair, q = finish
```
Output: `data/sessions/<name>/{session.json, frames.jsonl, rgb/*.png, depth/*.png}` (git-ignored). Depth is 16-bit PNG in millimetres, registered to the RGB image, 0 = invalid. Load with `kinect3dgs.sessions.load_session`. Stop with Ctrl-C (clean); `scripts/clean_shm.sh` removes leaked shared memory after a hard kill.
