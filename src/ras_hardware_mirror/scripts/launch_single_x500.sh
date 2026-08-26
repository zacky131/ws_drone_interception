#!/usr/bin/env bash
set -euo pipefail

workspace=${WS_DRONE_INTERCEPTION:-/home/zacky/ws_drone_interception}
if [[ ! -f "$workspace/scripts/env.sh" ]]; then
  echo "Cannot locate corrected environment at $workspace/scripts/env.sh" >&2
  exit 1
fi
source "$workspace/scripts/env.sh"

display=gui
while [[ $# -gt 0 ]]; do
  case "$1" in
    --headless) display=headless; shift ;;
    --gui) display=gui; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

px4_bin="$PX4_DIR/build/px4_sitl_default/bin/px4"
existing_px4=$(pgrep -f "$px4_bin" || true)
existing_gz=$(pgrep -f '^gz sim ' || true)
if [[ -n "$existing_px4" || -n "$existing_gz" ]]; then
  echo "PX4/Gazebo already running; stop that foreground session before launching another" >&2
  for pid in $existing_px4 $existing_gz; do
    ps -p "$pid" -o pid=,ppid=,stat=,args= >&2 || true
  done
  exit 1
fi

if ! pgrep -f '^MicroXRCEAgent udp4 -p 8888$' >/dev/null; then
  echo "NOTICE: MicroXRCEAgent is not running. Start it in another terminal for ROS/PX4 topics:" >&2
  echo "  MicroXRCEAgent udp4 -p 8888" >&2
fi

cd "$PX4_DIR"
echo "Starting the standard PX4 gz_x500 target in the foreground (world=default, instance=0)."
echo "Ctrl-C in this terminal stops PX4 and its Gazebo process tree."
interrupted=0
trap 'interrupted=1' INT TERM
set +e
if [[ "$display" == headless ]]; then
  env HEADLESS=1 make px4_sitl gz_x500
  status=$?
else
  env -u HEADLESS make px4_sitl gz_x500
  status=$?
fi
set -e
if [[ "$interrupted" == 1 || "$status" == 130 || "$status" == 143 ]]; then
  exit 0
fi
exit "$status"
