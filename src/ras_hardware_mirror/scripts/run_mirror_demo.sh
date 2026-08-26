#!/usr/bin/env bash
set -euo pipefail
method=M1
trajectory=HT1
condition=HC1
seed=1
repetition=1
stage=""
gui=--gui
output_root=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --method) method=$2; shift 2 ;;
    --trajectory) trajectory=$2; shift 2 ;;
    --condition) condition=$2; shift 2 ;;
    --seed) seed=$2; shift 2 ;;
    --repetition) repetition=$2; shift 2 ;;
    --stage) stage=$2; shift 2 ;;
    --headless) gui=--headless; shift ;;
    --output-root) output_root=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
visualization_only=false
case "$stage" in
  "") ;;
  GS0) visualization_only=true; trajectory=HT1; condition=DEV0; method=M1 ;;
  GS1) trajectory=STATIC; condition=DEV0; method=M1 ;;
  GS2) trajectory=HT1; condition=DEV0; method=M1 ;;
  GS3) trajectory=HT1; condition=HC0; method=M1 ;;
  GS4) trajectory=HT1; condition=HC1; method=M1 ;;
  GS5) trajectory=HT1; condition=HC1 ;;
  GS6)
    script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
    exec python3 -m pytest -q "$script_dir/../test/test_safety_logic.py"
    ;;
  *) echo "stage must be GS0..GS6" >&2; exit 2 ;;
esac
if [[ -n "$stage" && "$stage" != GS0 && "$stage" != GS6 && -z "$output_root" ]]; then
  script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
  workspace=${WS_DRONE_INTERCEPTION:-$(cd -- "$script_dir/../../.." && pwd)}
  output_root="$workspace/results/ras_hardware_mirror/development/$stage"
fi
if [[ "$stage" == GS5 ]]; then
  echo "GS5 is one arm at a time. Run this command once with --method M0prime and once with --method M1 using seed $seed."
fi
exec ros2 launch ras_hardware_mirror mirror_demo.launch.py method:="$method" trajectory:="$trajectory" condition:="$condition" seed:="$seed" repetition:="$repetition" visualization_only:="$visualization_only" output_root:="$output_root" gui:="$gui"
