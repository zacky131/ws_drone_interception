# Autonomous Drone Interception (ROS 2 Humble + PX4)

This repository contains the ROS 2 workspace for **Autonomous Quadrotor Target Interception**, featuring:
- **Model Predictive Control (MPC - M1)** with source-time delayed CA rollback/repropagation vs. **M0′ Baseline** (delayed single-CA update).
- **PX4 Autopilot Integration** via Micro XRCE-DDS (Offboard mode).
- **Deterministic Virtual Target & Telemetry Emulator** (controllable delays: DEV0 = 0 ms, HC0 = 50 ms, HC1 = 120 ms).
- **Safety State Machine & Dual Supervision** (geofence, timeout, manual override, auto-land).
- **Hardware-Mirror Architecture** for matched execution across **Gazebo Simulation** and **Real Hardware (Pixhawk / RTK-GNSS)**.

---

## 📁 Repository Structure

```text
ws_drone_interception/
├── src/
│   ├── ras_hardware_mirror/       # State providers, experiment manager, keyboard control, dashboard, launch files
│   ├── drone_interception_px4/    # Controller adapters, kinematics, baseline & MPC execution nodes
│   └── px4_msgs/                  # ROS 2 interface messages matching PX4 v1.15.2
├── vendor/
│   └── drone-interception-comparison-v3/  # Frozen MPC (ACADOS), target estimators, dynamic models
├── configs/                       # Controller parameters and scenario configs
├── deployment/                    # Environment verification scripts and package lockfiles
├── scripts/
│   └── env.sh                     # Workspace environment configuration script
├── requirements.txt               # Core Python dependencies
├── requirements-dev.txt           # Testing and formatting dependencies
└── requirements-px4.txt           # PX4/Micro-ROS Python dependencies
```

---

## 🛠️ Prerequisites & Installation

### 1. System Requirements
- **Ubuntu 22.04 LTS**
- **ROS 2 Humble** (Desktop Install)
- **PX4-Autopilot** (v1.14 or v1.15)
- **Gazebo Sim** (Harmonic / Garden / GZ Sim 8.x)
- **Micro XRCE-DDS Agent**
- **ACADOS** (with Python `acados_template` interface)
- **Python 3.10**

### 2. Clone and Setup Environment

```bash
cd /home/wens/ws_drone_interception

# Edit scripts/env.sh if your PX4 or ACADOS paths differ:
# Choose network mode:
# Option A (LOCAL MODE - default, immune to slow routers):
source scripts/env.sh local

# Option B (NETWORK MODE - enables remote RViz on ground laptop):
# source scripts/env.sh network
```

### 3. Build the Workspace

```bash
cd /home/wens/ws_drone_interception
source scripts/env.sh

# Clean Python path to prevent setuptools version collisions
clean_pythonpath=$(python3 - <<'PY'
import os
print(':'.join(p for p in os.environ.get('PYTHONPATH', '').split(':')
               if p and '/home/wens/.local/lib/python3.10/site-packages' not in p))
PY
)

PYTHONNOUSERSITE=1 PYTHONPATH="$clean_pythonpath" colcon build \
  --packages-select px4_msgs drone_interception_px4 ras_hardware_mirror

source install/setup.bash
```

### 4. Validate Environment

```bash
bash src/ras_hardware_mirror/scripts/validate_environment.sh
```
Ensure output finishes with: `ENVIRONMENT VALIDATION PASS`.

---

## 🚀 Execution Mode 1: Gazebo Simulation

In simulation, PX4 SITL and Gazebo run the X500 model. Open **6 separate terminals**:

```bash
# In EVERY terminal, first run:
cd /home/wens/ws_drone_interception
source scripts/env.sh
source install/setup.bash
```

### **Terminal 1: MicroXRCEAgent (SITL UDP Bridge)**
```bash
MicroXRCEAgent udp4 -p 8888
```

### **Terminal 2: PX4 SITL & Gazebo Simulator**
```bash
ros2 launch ras_hardware_mirror mirror_gazebo.launch.py gui:=--gui
```
*(Runs the equivalent of `make px4_sitl gz_x500` in the PX4 directory)*

### **Terminal 3: Experiment Nodes**
```bash
ros2 launch ras_hardware_mirror mirror_nodes.launch.py
```
> *Optional parameters:*
> - `method:=M1` or `method:=M0prime`
> - `trajectory:=HT1` or `trajectory:=STATIC` / `HT2`
> - `condition:=HC1` (120ms delay) or `HC0` (50ms) / `DEV0` (0ms)

### **Terminal 4: Interactive Keyboard Control & Flight Operator**
```bash
ros2 run ras_hardware_mirror keyboard_control
```
*Or for a specific trial from the 24-run manifest:*
```bash
ros2 launch ras_hardware_mirror mirror_keyboard.launch.py manifest_row:=1
```

### **Terminal 5: RViz2 3D Visualizer**
```bash
rviz2 -d src/ras_hardware_mirror/config/rviz_hardware_mirror.rviz
```

### **Terminal 6: Real-Time 4-Panel Operator Dashboard**
```bash
ros2 run ras_hardware_mirror live_dashboard --ros-args \
  -p config:=/home/wens/ws_drone_interception/src/ras_hardware_mirror/config/hardware_mirror_dev.yaml \
  -p field:=/home/wens/ws_drone_interception/src/ras_hardware_mirror/config/field.yaml
```

---

## 🛰️ Execution Mode 2: Real Hardware Implementation

For real hardware deployment (e.g., Holybro X500 / Pixhawk 6X / companion computer with RTK-GNSS):

### 1. Hardware & PX4 Pre-flight Setup
1. **Physical Connection**: Connect companion computer to Pixhawk via Serial (e.g. `TELEM2` port) or USB (`/dev/ttyACM0`) or Ethernet.
2. **PX4 Parameters (set via QGroundControl)**:
   - `UXRCE_DDS_CFG` = `TELEM2` (or corresponding port / Ethernet)
   - `SER_TEL2_BAUD` = `921600`
   - `COM_OF_LOSS_T` = `1.0` s (Offboard loss failsafe timeout)
   - `NAV_RCL_ACT` = `Hold` or `Return to Launch`
3. **RTK-GNSS**: Confirm PX4 EKF2 reports a solid 3D fix (`RTK Fixed` / Fix Type 6).
4. **Field Boundaries**: Update geofence, flight altitude, and origin in [`src/ras_hardware_mirror/config/field.yaml`](file:///home/wens/ws_drone_interception/src/ras_hardware_mirror/config/field.yaml) and [`src/ras_hardware_mirror/config/hardware_mirror_dev.yaml`](file:///home/wens/ws_drone_interception/src/ras_hardware_mirror/config/hardware_mirror_dev.yaml).

### 2. Automatic Field Orientation Calibration (Run Once at the Field)

Before starting the experiment nodes, place your drone on the takeoff pad **facing in the forward direction along the length of your football field**.

1. Start `MicroXRCEAgent` (Terminal 1) so ROS 2 can read the drone's compass heading.
2. Run the calibrator:
   ```bash
   ros2 run ras_hardware_mirror calibrate_field
   ```
   *Or with a manual compass heading (e.g. 45.0°):*
   ```bash
   ros2 run ras_hardware_mirror calibrate_field --heading 45.0
   ```
This automatically samples the drone's compass heading, calculates the rotated ENU field geometry, and updates `field.yaml` and `hardware_mirror_dev.yaml` so the pursuer, target trajectory, and geofence align with your physical football pitch.

---

### 3. Terminal-by-Terminal Execution for Real Hardware

```bash
# In EVERY terminal, first run:
cd /home/wens/ws_drone_interception
source scripts/env.sh
source install/setup.bash
```

#### **Terminal 1: MicroXRCEAgent (Physical Serial / UART / Ethernet Bridge)**
```bash
# For USB/Serial (Pixhawk TELEM2 / USB port):
MicroXRCEAgent serial --dev /dev/ttyACM0 -b 921600

# (Or for UART port /dev/ttyUSB0, /dev/ttyTHS1):
# MicroXRCEAgent serial --dev /dev/ttyUSB0 -b 921600

# (Or for Ethernet connection):
# MicroXRCEAgent udp4 -p 8888
```
> **Verification**: Check topic stream in another terminal: `ros2 topic list | grep fmu`. You should see `/fmu/out/vehicle_status` and `/fmu/out/vehicle_local_position`.

#### **Terminal 2: (OMITTED ON REAL HARDWARE)**
> ⚠️ **Do NOT run Gazebo or `mirror_gazebo.launch.py`.** The physical drone running PX4 replaces the simulation terminal.

#### **Terminal 3: Experiment Nodes**
```bash
ros2 launch ras_hardware_mirror mirror_nodes.launch.py
```
> *Example with specific experiment parameters:*
> ```bash
> ros2 launch ras_hardware_mirror mirror_nodes.launch.py method:=M1 trajectory:=HT1 condition:=HC1
> ```

#### **Terminal 4: Interactive Keyboard Control (Operator Station)**
```bash
ros2 run ras_hardware_mirror keyboard_control
```
*Or, if running a numbered row from the 24-run manifest:*
```bash
ros2 launch ras_hardware_mirror mirror_keyboard.launch.py manifest_row:=1
```

#### **Terminal 5: RViz2 (Ground Station GCS)**
```bash
rviz2 -d src/ras_hardware_mirror/config/rviz_hardware_mirror.rviz
```

#### **Terminal 6: Live 4-Panel Dashboard (Ground Station GCS)**
```bash
ros2 run ras_hardware_mirror live_dashboard --ros-args \
  -p config:=/home/wens/ws_drone_interception/src/ras_hardware_mirror/config/hardware_mirror_dev.yaml \
  -p field:=/home/wens/ws_drone_interception/src/ras_hardware_mirror/config/field.yaml
```

---

## 🎮 Flight Operator Key Bindings (Terminal 4)

| Key | Action / Stage | Description |
|---|---|---|
| **`q`** | **Arm & Offboard** | Prestreams zero-velocity setpoints, switches PX4 to Offboard, and arms motors |
| **`t`** | **Manual Takeoff** | Takes off to safe 2.0 m altitude and enters stable hover |
| **`g`** | **Land** | Cancels active pursuit / trajectory and commands immediate landing |
| **`1`** | **GS1** | Static target, DEV0 (0 ms delay), M1 MPC |
| **`2`** | **GS2** | HT1 trajectory, DEV0 (0 ms delay), M1 MPC |
| **`3`** | **GS3** | HT1 trajectory, HC0 (50 ms delay), M1 MPC |
| **`4`** | **GS4** | HT1 trajectory, HC1 (120 ms delay), M1 MPC |
| **`5`** | **GS5 (M0′)** | HT1 / HC1 (120 ms delay) / M0′ Baseline, paired seed 6505 |
| **`6`** | **GS5 (M1)** | HT1 / HC1 (120 ms delay) / M1 MPC, paired seed 6505 |
| **`8`** | **Manifest Row** | Executes the selected row from `hardware_mirror_24.csv` |
| **`w` / `s`** | **Z-Velocity** | Hold to move Up / Down (manual override) |
| **`a` / `d`** | **Yaw Rate** | Hold to turn Left / Right |
| **`↑` / `↓` / `←` / `→`** | **Horizontal** | Hold to move Forward / Back / Left / Right (triggers instant abort if in `RUN`) |

### Safety Notes for Hardware Flights:
1. **RC Transmitter Priority**: Always keep your RC transmitter powered on with a physical flight-mode switch configured to override Offboard mode to Position/Manual or Kill Switch at any moment.
2. **Propeller Safety**: Perform first-time communications and offboard engagement on the bench **with propellers removed**.
3. **Emergency Takeover**: Touching any movement key (`w/s/a/d/arrows`) during an autonomous pursuit instantly transitions `RUN -> ABORT -> HOLD` for manual repositioning.

---

## 📊 Results & Analysis

Experiment logs are saved per run to `results/ras_hardware_mirror/<trajectory>_<condition>_<method>_repNN/`:
- `steps.csv`: Interceptor p/v/a, virtual target truth/estimate, separation distance, MPC horizon.
- `telemetry_packets.csv`: Packet IDs, source timestamps, arrival timestamps, delay error.
- `experiment_events.csv`: Exact timestamps of state machine transitions.
- `metadata.json`: Run status (`"complete": true`), configuration snapshot, and hashes.

To run matched pair analysis comparing M0′ vs. M1:
```bash
python3 src/ras_hardware_mirror/scripts/analyze_mirror_pair.py \
  results/ras_hardware_mirror/HT1_HC1_M0prime_rep01 \
  results/ras_hardware_mirror/HT1_HC1_M1_rep01 \
  --output results/ras_hardware_mirror/analysis_pair_seed6505
```
