#!/usr/bin/env bash
set -euo pipefail
method=${1:-M1}
trajectory=${2:-HT1}
condition=${3:-HC1}
seed=${4:-1}
repetition=${5:-1}
exec ros2 launch ras_hardware_mirror mirror_nodes.launch.py method:="$method" trajectory:="$trajectory" condition:="$condition" seed:="$seed" repetition:="$repetition"
