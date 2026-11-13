#!/usr/bin/env bash
# One-time system setup for the Kinect. Usage: system_setup.sh [--check|--uninstall]  (apply needs sudo)
set -u
source "$(dirname "$0")/env.sh"
RULE_SRC="$SRC_DIR/platform/linux/udev/51-kinect.rules"
RULE_DST="$ETC_ROOT/udev/rules.d/51-kinect.rules"
BL_DST="$ETC_ROOT/modprobe.d/kinect3dgs-blacklist.conf"
BL_BODY=$'blacklist gspca_kinect\ninstall gspca_kinect /bin/false'
mode="${1:-apply}"

st_rule()   { [ -f "$RULE_DST" ] && cmp -s "$RULE_SRC" "$RULE_DST"; }
st_bl()     { [ -f "$BL_DST" ] && grep -qx 'blacklist gspca_kinect' "$BL_DST" && grep -qx 'install gspca_kinect /bin/false' "$BL_DST"; }
st_module() { ! lsmod 2>/dev/null | grep -q '^gspca_kinect'; }
report() { local ok=0
  for pair in "udev rule:st_rule" "modprobe blacklist:st_bl" "gspca_kinect unloaded:st_module"; do
    if "${pair#*:}"; then echo "OK       ${pair%%:*}"; else echo "MISSING  ${pair%%:*}"; ok=1; fi
  done; return $ok; }

case "$mode" in
  --check) report; exit $? ;;
  --uninstall)
    [ "$(id -u)" = 0 ] || { echo "re-run with sudo"; exit 1; }
    rm -f "$RULE_DST" "$BL_DST"; udevadm control --reload-rules; echo "removed"; exit 0 ;;
  apply)
    [ "$(id -u)" = 0 ] || { echo "re-run with sudo: sudo bash $0"; exit 1; }
    [ -f "$RULE_SRC" ] || { echo "run build_libfreenect.sh first (need $RULE_SRC)"; exit 1; }
    st_rule || install -Dm0644 "$RULE_SRC" "$RULE_DST"
    st_bl   || { mkdir -p "$(dirname "$BL_DST")"; printf '%s\n' "$BL_BODY" > "$BL_DST"; }
    st_module || modprobe -r gspca_kinect || true
    udevadm control --reload-rules && udevadm trigger --subsystem-match=usb --action=add
    report; echo "Now unplug and replug the Kinect USB cable." ;;
  *) echo "usage: $0 [--check|--uninstall]"; exit 2 ;;
esac
