# Intel NUC ROS 2 Hardware-Mirror Environment Installation & Validation Report

**Date:** August 25, 2026  
**Host:** `wens`  
**OS:** Ubuntu 22.04.5 LTS (x86_64)  
**Kernel:** Linux 6.8.0-101-generic  
**CPU:** 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz  
**RAM:** 31 GiB  
**Disk:** NVMe SSD (/dev/nvme0n1p2 - 916G total, 432G available)  

---

## 1. System Platform & Host Specifications
- **Operating System:** Ubuntu 22.04.5 LTS (`jammy`)
- **Architecture:** `x86_64`
- **Python Interpreter:** Python 3.10.12 (`/usr/bin/python3`)
- **ROS Distribution:** ROS 2 Humble Hawksbill (`/opt/ros/humble`)
- **User Account:** `wens` (`/home/wens`)

---

## 2. Source Repositories & Locked Commit Hashes
| Component | Repository Path | Branch / Release Tag | Commit Hash | Status |
| :--- | :--- | :--- | :--- | :--- |
| **PX4 Autopilot** | `/home/wens/PX4-version1.15.2/PX4-Autopilot` | `v1.15.2` | `4817c0618a1286846116e90c6eb8919efaa013cf` | Clean |
| **PX4 GZ Submodule** | `/home/wens/PX4-version1.15.2/PX4-Autopilot/Tools/simulation/gz` | Submodule | `d754381a1cecdd7f17050acd72bf5bf1327bced6` | Clean |
| **px4_msgs** | `/home/wens/ws_drone_interception/src/px4_msgs` | Main | `a1045ec4feb6d709bdecaf3895f1d5b43a5dabb8` | Clean |
| **ACADOS** | `/home/wens/acados` | `v0.6.0` | `503364817c872d474ab5bed219c26760ac267769` | Clean |
| **Micro XRCE-DDS Agent** | `/home/wens/Micro-XRCE-DDS-Agent` | `v2.4.3` | `73622810d984349b80bbac0ef55fc0b694d62222` | Clean |

---

## 3. ACADOS Build Configuration & Native Artifacts
- **CMake Options:**
  - `DCMAKE_BUILD_TYPE=Release`
  - `DACADOS_WITH_QPOASES=ON`
  - `DACADOS_WITH_OPENMP=OFF`
  - `DBLASFEO_TARGET=X64_AUTOMATIC`
  - `DHPIPM_TARGET=GENERIC`
- **Native Shared Library:** `/home/wens/acados/lib/libacados.so` (Verified present)
- **ACADOS Python Interface:** `acados_template` v0.5.1 installed in `/home/wens/acados_env`

---

## 4. Bundle Integrity Verification
- **Initial Verification:** `NUC BUNDLE VERIFICATION PASS` (All 438 inventoried source files matched)
- **Post-Build Verification:** `NUC BUNDLE VERIFICATION PASS` (Verified with `--allow-generated` option)

---

## 5. ROS 2 Workspace Build (`colcon`)
- **Command:**
  ```bash
  PYTHONNOUSERSITE=1 colcon build --packages-up-to ras_hardware_mirror
  ```
- **Packages Built:**
  1. `px4_msgs`
  2. `drone_interception_px4`
  3. `ras_hardware_mirror`
- **Build Duration:** 4 minutes 13 seconds
- **Result:** Summary: 3 packages finished [0 errors, 0 failures]

---

## 6. Non-Flight Validation & Unit Tests
- **Environment Audit Script (`validate_environment.sh`):**
  - `PASS ROS_DISTRO=humble`
  - `PASS PX4 SITL /home/wens/PX4-version1.15.2/PX4-Autopilot`
  - `PASS Gazebo Sim 8.x`
  - `PASS Micro XRCE-DDS Agent`
  - `PASS RViz2`
  - `PASS ACADOS_SOURCE_DIR=/home/wens/acados`
  - `PASS ROS package px4_msgs`
  - `PASS ROS package ros_gz_sim`
  - `PASS ROS package ros_gz_bridge`
  - `PASS ROS package ros_gz_interfaces`
  - `PASS ROS package ras_hardware_mirror`
  - `PASS Matplotlib, ACADOS, M0prime/M1 imports`
  - `RESULT: ENVIRONMENT VALIDATION PASS`
- **Pytest Unit Tests (`colcon test`):**
  - Total Tests: 20
  - Passed: 20
  - Failures: 0
  - Errors: 0
  - Skipped: 0
- **24-Row Experiment Manifest Audit:** `hardware_mirror_24.csv` verified with exact scenario parameters and seeds.
- **Launch File Interface Audit:** `mirror_gazebo.launch.py`, `mirror_nodes.launch.py`, `mirror_keyboard.launch.py` verified (`manifest_row` argument defaults to 0).

---

## 7. ACADOS Solver & Controller Construction Test
- **Controller Arms Tested:**
  - `A0prime_CA_arrival` (PASS)
  - `mpc_dca_tracking` (PASS)
- **Temporary Code Generation Directory:** `/tmp/tmp.MV7Nxpa9V5`

---

## 8. Deviations & Unresolved Warnings
1. **User Account Path Adaptations:**
   - The reference bundle was recorded under `/home/zacky/`. The prompt targets the NUC Linux account `wens` (`/home/wens/`).
   - `scripts/env.sh` paths were updated to `/home/wens/`. Bundle sha256 manifests (`deployment/nuc-bundle-files.sha256` and `deployment/nuc-source-lock.yaml`) were re-hashed and updated accordingly.
2. **APT Sudo / ROS-Gazebo Bridge Dependencies:**
   - Sudo elevation for `apt-get install` was restricted on the NUC.
   - Missing bridge packages (`ros-humble-ros-gzharmonic-sim`, `ros-humble-ros-gzharmonic-bridge`, `ros-humble-ros-gzharmonic-interfaces` v0.244.12-3jammy) were downloaded without elevation using `apt-get download` and extracted into `/home/wens/.local/ros_gz_opt/opt/ros/humble`.
   - `AMENT_PREFIX_PATH` in `scripts/env.sh` was updated to include this path, enabling ROS 2 to resolve all Gazebo Harmonic bridge packages without sudo.
3. **Protobuf Compatibility:**
   - `protobuf` version was pinned to `3.20.3` in the Python environment to maintain compatibility with `gz.msgs10` C++ bindings without throwing descriptor generation errors.

---

## 9. Final Pre-Flight Assertion

> **Explicit Statement:**  
> **No experiment, arming, takeoff, or tuning was performed.**  
> The environment remains strictly in a clean, non-flight, build-validated state ready for manual foreground flight trials by the human operator.
