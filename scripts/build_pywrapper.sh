#!/usr/bin/env bash
# Build upstream freenect.pyx against the local prefix and install into the venv.
set -euo pipefail
source "$(dirname "$0")/env.sh"
[ -f "$PREFIX/lib/libfreenect_sync.so" ] || { echo "run build_libfreenect.sh first"; exit 1; }
cd "$ROOT"
[ -d .venv ] || uv sync --inexact
cp "$SRC_DIR/wrappers/python/freenect.pyx" pywrapper/freenect.pyx
mkdir -p build && sha256sum pywrapper/freenect.pyx > build/pywrapper.sha256
KINECT3DGS_PREFIX="$PREFIX" uv pip install --reinstall ./pywrapper
so=$(uv run python -c "import freenect; print(freenect.__file__)")
echo "$so"
env -u LD_LIBRARY_PATH ldd "$so" | grep -E "libfreenect" 
