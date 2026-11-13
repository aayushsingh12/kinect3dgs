#!/usr/bin/env bash
# List USB devices from sysfs (no usbutils needed) and flag Kinect IDs.
set -u
source "$(dirname "$0")/env.sh"
camera=0
for d in "$SYSFS_USB"/*; do
  [ -f "$d/idVendor" ] || continue
  vid=$(<"$d/idVendor"); pid=$(<"$d/idProduct")
  mfr=$(cat "$d/manufacturer" 2>/dev/null || true); prod=$(cat "$d/product" 2>/dev/null || true)
  speed=$(cat "$d/speed" 2>/dev/null || echo "?")
  mark=""
  if [ "$vid" = "045e" ]; then
    case "$pid" in
      02ae|02bf) mark="KINECT camera"; camera=1 ;;
      02ad|02be) mark="KINECT audio" ;;
      02b0)      mark="KINECT motor" ;;
      02c2)      mark="KINECT hub/motor (K4W/1473)" ;;
    esac
  fi
  printf '%s %s:%s %s | %s | %s Mb/s  %s\n' "$(basename "$d")" "$vid" "$pid" "$mfr" "$prod" "$speed" "$mark"
done
if [ "$camera" = 1 ]; then echo "camera: FOUND"; exit 0
else echo "camera: MISSING (check 12V adapter, cable, port)"; exit 1; fi
