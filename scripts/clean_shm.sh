#!/usr/bin/env bash
# Remove leaked kinect3dgs shared-memory segments whose creator process is gone.
for f in /dev/shm/kinect3dgs-*; do
  [ -e "$f" ] || continue
  pid=$(basename "$f" | sed -E 's/^kinect3dgs-([0-9]+)-.*/\1/')
  if ! kill -0 "$pid" 2>/dev/null; then rm -f "$f" && echo "removed $f"; fi
done
