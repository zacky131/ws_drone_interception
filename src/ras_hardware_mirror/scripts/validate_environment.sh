#!/usr/bin/env bash
set -uo pipefail
failures=0
pass() { echo "PASS  $*"; }
fail() { echo "FAIL  $*" >&2; failures=$((failures + 1)); }
[[ "${ROS_DISTRO:-}" == humble ]] && pass "ROS_DISTRO=humble" || fail "source scripts/env.sh (expected ROS_DISTRO=humble)"
[[ -n "${PX4_DIR:-}" && -x "${PX4_DIR:-/missing}/build/px4_sitl_default/bin/px4" ]] && pass "PX4 SITL ${PX4_DIR}" || fail "PX4_DIR/SITL binary unavailable"
if command -v gz >/dev/null && gz sim --version 2>&1 | grep -E 'version 8\.' >/dev/null; then pass "Gazebo Sim 8.x"; else fail "Gazebo Sim 8.x unavailable"; fi
command -v MicroXRCEAgent >/dev/null && pass "Micro XRCE-DDS Agent" || fail "MicroXRCEAgent unavailable"
command -v rviz2 >/dev/null && pass "RViz2" || fail "RViz2 unavailable"
[[ -n "${ACADOS_SOURCE_DIR:-}" && -d "${ACADOS_SOURCE_DIR:-/missing}" ]] && pass "ACADOS_SOURCE_DIR=${ACADOS_SOURCE_DIR}" || fail "ACADOS environment unavailable"
for package in px4_msgs ros_gz_sim ros_gz_bridge ros_gz_interfaces ras_hardware_mirror; do
  ros2 pkg prefix "$package" >/dev/null 2>&1 && pass "ROS package $package" || fail "ROS package $package unresolved"
done
python3 - <<'PY' >/tmp/ras_hw_mirror_import_check.txt 2>&1
import matplotlib, acados_template
from gz.transport13 import Node as GazeboTransportNode
from gz.msgs10.pose_pb2 import Pose as GazeboPose
from m0prime_confirmatory.controller import ConfirmatoryControllerAdapter, METHODS
assert METHODS == ("A0prime_CA_arrival", "mpc_dca_tracking")
print("controller mapping", METHODS)
PY
if [[ $? == 0 ]]; then pass "Matplotlib, ACADOS, M0prime/M1 imports"; else fail "Python/controller import failed: $(tr '\n' ' ' </tmp/ras_hw_mirror_import_check.txt)"; fi
rm -f /tmp/ras_hw_mirror_import_check.txt
if [[ $failures -ne 0 ]]; then echo "ENVIRONMENT VALIDATION FAILED ($failures checks)" >&2; exit 1; fi
echo "ENVIRONMENT VALIDATION PASS"
