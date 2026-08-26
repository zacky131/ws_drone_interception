#!/usr/bin/env bash
set -euo pipefail

workspace=${WS_DRONE_INTERCEPTION:-/home/zacky/ws_drone_interception}
source "$workspace/scripts/env.sh"
px4_bin="$PX4_DIR/build/px4_sitl_default/bin/px4"
mapfile -t px4_pids < <(pgrep -f "^$px4_bin( |$)" || true)
mapfile -t gz_pids < <(pgrep -f '^gz sim ' || true)

if [[ ${#px4_pids[@]} -eq 0 && ${#gz_pids[@]} -eq 0 ]]; then
  echo "No foreground PX4 gz_x500 session found."
  exit 0
fi

for pid in "${px4_pids[@]}" "${gz_pids[@]}"; do
  [[ -n "$pid" ]] || continue
  ps -p "$pid" -o pid=,ppid=,stat=,args=
done
for pid in "${px4_pids[@]}" "${gz_pids[@]}"; do
  [[ -n "$pid" ]] || continue
  kill -TERM "$pid" 2>/dev/null || true
done
for _ in $(seq 1 50); do
  alive=0
  for pid in "${px4_pids[@]}" "${gz_pids[@]}"; do
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && alive=1
  done
  [[ "$alive" == 0 ]] && break
  sleep 0.1
done
for pid in "${px4_pids[@]}" "${gz_pids[@]}"; do
  [[ -n "$pid" ]] || continue
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL "$pid"
  fi
done
echo "Stopped the listed PX4/Gazebo processes."
