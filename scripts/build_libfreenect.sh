#!/usr/bin/env bash
# Build libfreenect (pinned tag) into the project-local prefix. No sudo.
set -euo pipefail
source "$(dirname "$0")/env.sh"
for c in git cmake make cc pkg-config; do command -v "$c" >/dev/null || { echo "missing: $c"; exit 1; }; done
pkg-config --exists libusb-1.0 || { echo "missing libusb-1.0 dev files"; exit 1; }

[ -d "$SRC_DIR" ] || git clone --branch "$FREENECT_TAG" --depth 1 "$FREENECT_URL" "$SRC_DIR"
actual=$(git -C "$SRC_DIR" rev-parse HEAD)
echo "libfreenect $FREENECT_TAG = $actual"
apply_patches
if [ -f "$PREFIX/lib/libfreenect.so" ] && [ "$(cat "$PREFIX/.installed-tag" 2>/dev/null)" = "$INSTALL_MARK" ]; then
  echo "libfreenect $INSTALL_MARK already installed in $PREFIX"; exit 0
fi
if [ -n "$FREENECT_COMMIT" ] && [ "$actual" != "$FREENECT_COMMIT" ]; then echo "tag moved! refusing"; exit 1; fi

configure() {  # $1 = BUILD_EXAMPLES value, rest = extra flags
  local ex="$1"; shift
  cmake -S "$SRC_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" -DPROJECT_INCLUDE_INSTALL_DIR="$PREFIX/include" \
    -DBUILD_REDIST_PACKAGE=ON -DBUILD_EXAMPLES="$ex" -DBUILD_C_SYNC=ON \
    -DBUILD_PYTHON3=OFF -DBUILD_PYTHON2=OFF -DBUILD_OPENNI2_DRIVER=OFF \
    -DBUILD_CV=OFF -DBUILD_AS3_SERVER=OFF -DBUILD_FAKENECT=OFF \
    -DOpenGL_GL_PREFERENCE=GLVND -Wno-dev "$@"
}
if ! configure ON; then
  echo ">> configure failed; retrying with CMake policy workaround"
  configure ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5
fi
if ! cmake --build "$BUILD_DIR" -j"$(nproc)"; then
  echo ">> build failed with examples; retrying without examples"
  configure OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5
  cmake --build "$BUILD_DIR" -j"$(nproc)"
  mkdir -p "$PREFIX/bin"
  cc "$SRC_DIR/examples/camtest.c" -I"$PREFIX/include" -I/usr/include/libusb-1.0 \
     -L"$BUILD_DIR/lib" -lfreenect -o "$BUILD_DIR/freenect-camtest" || true
fi
cmake --install "$BUILD_DIR"
echo "$INSTALL_MARK" > "$PREFIX/.installed-tag"
test -f "$PREFIX/lib/libfreenect.so" && test -f "$PREFIX/include/libfreenect/libfreenect_sync.h" && echo "OK"
