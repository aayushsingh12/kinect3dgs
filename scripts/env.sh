# Shared paths and pins. Source this; do not execute.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FREENECT_TAG="v0.7.5"
FREENECT_COMMIT="09a1f098040d00e6070c18174904547ec31d2774"
FREENECT_URL="https://github.com/OpenKinect/libfreenect"
PREFIX="$ROOT/.local"
SRC_DIR="$ROOT/third_party/libfreenect"
BUILD_DIR="$ROOT/build/libfreenect"
SYSFS_USB="${KINECT3DGS_SYSFS_USB:-/sys/bus/usb/devices}"
ETC_ROOT="${KINECT3DGS_ETC:-/etc}"
export LD_LIBRARY_PATH="$PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

PATCH_HASH="$(cat "$ROOT"/patches/*.patch 2>/dev/null | sha256sum | cut -c1-12)"
INSTALL_MARK="${FREENECT_TAG}-${PATCH_HASH}"

# Apply local patches to the libfreenect checkout (idempotent).
apply_patches() {
  local patch
  for patch in "$ROOT"/patches/*.patch; do
    [ -f "$patch" ] || continue
    if git -C "$SRC_DIR" apply --check -R "$patch" 2>/dev/null; then echo "patch already applied: $(basename "$patch")"
    else git -C "$SRC_DIR" apply "$patch" && echo "applied $(basename "$patch")"; fi
  done
}
