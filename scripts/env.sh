#!/usr/bin/env bash
# Source this file from a fresh shell before running the campaign.

_ral_restore_nounset=0
if [[ $- == *u* ]]; then
  _ral_restore_nounset=1
  set +u
fi

export PX4_DIR=/home/wens/PX4-version1.15.2/PX4-Autopilot
export WS_DRONE_INTERCEPTION=/home/wens/ws_drone_interception
export DRONE_INTERCEPTION_V3="$WS_DRONE_INTERCEPTION/vendor/drone-interception-comparison-v3"
export ACADOS_SOURCE_DIR=/home/wens/acados

source /opt/ros/humble/setup.bash
source /home/wens/acados_env/bin/activate
if [[ -f "$WS_DRONE_INTERCEPTION/install/setup.bash" ]]; then
  source "$WS_DRONE_INTERCEPTION/install/setup.bash"
fi
export PYTHONPATH="$WS_DRONE_INTERCEPTION/install/drone_interception_px4/lib/python3.10/site-packages:$ACADOS_SOURCE_DIR/interfaces/acados_template:/home/wens/.local/lib/python3.10/site-packages:/usr/lib/python3/dist-packages${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export AMENT_PREFIX_PATH="/home/wens/.local/ros_gz_opt/opt/ros/humble:$WS_DRONE_INTERCEPTION/install/drone_interception_px4${AMENT_PREFIX_PATH:+:$AMENT_PREFIX_PATH}"
export PATH="$WS_DRONE_INTERCEPTION/install/drone_interception_px4/lib/drone_interception_px4:$PATH"
export GZ_IP=127.0.0.1
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

# Network mode selection: 'local' (default, immune to slow routers) or 'network' (multi-machine RViz)
_net_arg="${1:-${ROS_NET_MODE:-local}}"
if [[ "$_net_arg" == "network" || "$_net_arg" == "net" || "$_net_arg" == "remote" || "$ROS_LOCALHOST_ONLY" == "0" && "$_net_arg" != "local" ]]; then
  export ROS_NET_MODE=network
  export ROS_LOCALHOST_ONLY=0
  if [[ -f "$WS_DRONE_INTERCEPTION/src/ras_hardware_mirror/config/fastdds_shm_network.xml" ]]; then
    export FASTRTPS_DEFAULT_PROFILES_FILE="$WS_DRONE_INTERCEPTION/src/ras_hardware_mirror/config/fastdds_shm_network.xml"
  fi
  echo "[ENV] ROS 2 Network Mode: NETWORK / REMOTE RVIZ (Domain: $ROS_DOMAIN_ID, Fast-DDS SHM+UDP)"
else
  export ROS_NET_MODE=local
  export ROS_LOCALHOST_ONLY=1
  unset FASTRTPS_DEFAULT_PROFILES_FILE
  echo "[ENV] ROS 2 Network Mode: LOCALHOST ONLY (Immune to slow routers)"
fi

if [[ "$_ral_restore_nounset" == 1 ]]; then
  set -u
fi
unset _ral_restore_nounset
unset _net_arg
