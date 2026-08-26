# RAS one-X500 hardware-mirror rehearsal

## A. Purpose

This package runs one PX4 SITL X500 interceptor against a deterministic software virtual target in a common `map`/ENU frame. It rehearses the later RTK/ENU architecture, the matched M0′ versus M1 timing comparison, the hardware-like experiment/safety state machine, Gazebo, RViz2, a live four-panel dashboard, and per-run logging.

This is a simulation rehearsal for the later one-X500 + RTK-GNSS + virtual-target outdoor experiment. It is not hardware validation and it adds no scientific method. The development field, altitude, motion, and capture settings are placeholders, not approved outdoor-flight values.

The repository-confirmed mapping is:

| Public method | Existing code ID | Timing mechanism |
|---|---|---|
| `M0prime` | `A0prime_CA_arrival` | delayed single-CA update at arrival time |
| `M1` | `mpc_dca_tracking` | source-time delayed single-CA rollback/repropagation |

Both arms import `ConfirmatoryControllerAdapter`, the same target predictor, and the same frozen 0.4-s / N=20 tracking-MPC implementation. The common PX4 odometry adapter forwards position, velocity, and PX4-reported acceleration as the existing controller's required nine-state pursuer vector. Unknown methods are rejected. M0, M2, M3, FRPN, and GPN are not exposed by this package. The manager applies the same 3 m/s² development transport limiter and 5 m/s speed envelope to either method; this does not retune the imported estimator or MPC.

## B. Package architecture

```text
VirtualTarget -> TelemetryEmulator -> existing M0′/M1 estimator-controller
                                      -> PX4 Offboard -> one X500 SITL

Gazebo + PX4 -> GazeboStateProvider -> common map/ENU NavigationState
ExperimentManager + SafetySupervisor -> state machine and independent abort
VisualizationNode -> RViz markers + minimal TF
LiveDashboard -> live ROS subscriptions (never CSV polling)
Logger -> metadata, snapshots, packet CSV, event CSV, and step CSV
```

`state_provider_interface.py` defines the provider boundary. `GazeboStateProviderNode` is the implemented simulation provider. `RTKStateProvider` is deliberately only a stub: no receiver driver or physical-site transform is claimed.

The scientific public topics are:

| Topic | Type |
|---|---|
| `/ras_hw_mirror/interceptor/state/ground_truth` | `nav_msgs/Odometry` |
| `/ras_hw_mirror/interceptor/state/px4` | `nav_msgs/Odometry` |
| `/ras_hw_mirror/target/truth` | `nav_msgs/Odometry` |
| `/ras_hw_mirror/target/measurement` | `geometry_msgs/PoseWithCovarianceStamped` |
| `/ras_hw_mirror/target/estimate` | `nav_msgs/Odometry` |
| `/ras_hw_mirror/target/prediction_path` | `nav_msgs/Path` |
| `/ras_hw_mirror/target/truth_path` | `nav_msgs/Path` |
| `/ras_hw_mirror/interceptor/path` | `nav_msgs/Path` |
| `/ras_hw_mirror/experiment/status` | `diagnostic_msgs/DiagnosticArray` |
| `/ras_hw_mirror/safety/status` | `diagnostic_msgs/DiagnosticArray` |
| `/ras_hw_mirror/telemetry/status` | `diagnostic_msgs/DiagnosticArray` |
| `/ras_hw_mirror/visualization/markers` | `visualization_msgs/MarkerArray` |
| `/ras_hw_mirror/safety/manual_abort` | `std_msgs/Bool` |
| `/ras_hw_mirror/safety/abort` | `std_msgs/Bool` |

Internal adapter topics are `/ras_hw_mirror/internal/target/measurement_state`, `/ras_hw_mirror/controller/command_raw`, `/ras_hw_mirror/controller/command_applied`, `/ras_hw_mirror/controller/status`, `/ras_hw_mirror/experiment/phase`, `/ras_hw_mirror/experiment/epoch`, and readiness heartbeats below `/ras_hw_mirror/ready/*`.

## C. Build from a clean terminal

The existing `drone_interception_px4` package must be rebuilt once because its current installed snapshot predates the M0′ Python module.

```bash
cd /home/wens/ws_drone_interception
source scripts/env.sh

# ROS Humble's colcon/setuptools path must not see the user-site setuptools 82.
clean_pythonpath=$(python3 - <<'PY'
import os
print(':'.join(p for p in os.environ.get('PYTHONPATH', '').split(':')
               if p and '/home/wens/.local/lib/python3.10/site-packages' not in p))
PY
)
PYTHONNOUSERSITE=1 PYTHONPATH="$clean_pythonpath" colcon build \
  --packages-select drone_interception_px4 ras_hardware_mirror

source install/setup.bash
```

The package reuses the corrected-Rev6 environment at `scripts/env.sh`, PX4 at `/home/wens/PX4-version1.15.2/PX4-Autopilot`, and the frozen controller configuration already used by the geometry campaign. It does not modify those inputs.

## D. Validate the environment

```bash
cd /home/wens/ws_drone_interception
source scripts/env.sh
source install/setup.bash

bash src/ras_hardware_mirror/scripts/validate_environment.sh
```

The final line must be `ENVIRONMENT VALIDATION PASS`. The validator checks ROS Humble, the PX4 SITL binary, Gazebo Sim 8.x, Micro XRCE-DDS, ACADOS, `px4_msgs`, `ros_gz`, RViz2, Matplotlib, and the exact two-method import/mapping. It changes nothing.

Run the package tests through colcon's pytest backend (the installed colcon default is `unittest`, while this package uses normal pytest functions):

```bash
colcon test --packages-select ras_hardware_mirror --python-testing pytest
colcon test-result --verbose
```

## E. Standard PX4 Gazebo world

The simulator now uses the exact standard PX4 target:

```bash
cd /home/wens/PX4-version1.15.2/PX4-Autopilot
make px4_sitl gz_x500
```

This is Gazebo world `default`, PX4 instance 0, system ID 1, model `x500_0`, and un-namespaced `/fmu/...` ROS topics. The package no longer starts the custom football-field SDF. Hard-geofence and target-region geometry remain visible in RViz and the dashboard.

The software target is inserted into the default Gazebo world at runtime as a static visual-only sphere with no collision or dynamics. Its movement uses `/world/default/set_pose`; X500 ground truth comes from `/world/default/dynamic_pose/info`. The 50-Hz target truth and controller do not depend on the Gazebo pose-update call.

## F. One-command visual demo

After the build and environment validation:

```bash
bash src/ras_hardware_mirror/scripts/run_mirror_demo.sh
```

This starts the standard PX4 default world and X500, experiment nodes, terminal
keyboard controller, RViz2, and dashboard as foreground children of `ros2
launch`. It does **not** start MicroXRCEAgent. Start the agent manually first.
The multi-terminal procedure below is recommended for debugging.

## G. Manual multi-terminal launch

Run the build/source preamble in every ROS terminal:

```bash
cd /home/wens/ws_drone_interception
source scripts/env.sh
source install/setup.bash
```

Terminal 1 — MicroXRCEAgent, kept visible in the foreground:

```bash
MicroXRCEAgent udp4 -p 8888
```

Terminal 2 — standard PX4 default world and one X500:

```bash
ros2 launch ras_hardware_mirror mirror_gazebo.launch.py gui:=--gui
```


This executes the equivalent of `make px4_sitl gz_x500` in the PX4 directory. Do not close this terminal; Ctrl-C stops the simulator tree.

Terminal 3 — experiment nodes:

```bash
# Terminal 3
cd ~/ws_drone_interception
source scripts/env.sh
source install/setup.bash
ros2 launch ras_hardware_mirror mirror_nodes.launch.py
```

Terminal 4 — keyboard control:

```bash
# Terminal 4
cd ~/ws_drone_interception
source scripts/env.sh
source install/setup.bash
ros2 run ras_hardware_mirror keyboard_control
```

Run this directly in its own interactive SSH terminal. If the SSH client does
not allocate a pseudo-terminal automatically, connect with `ssh -t`. No desktop,
X11 forwarding, or Tk window is used.


Alternative Terminal 5&6 — live dashboard:

```bash
cd /home/wens/ws_drone_interception
source scripts/env.sh
source install/setup.bash

ros2 launch ras_hardware_mirror mirror_visualization.launch.py
```

Terminal 5 — RViz:

```bash
rviz2 -d src/ras_hardware_mirror/config/rviz_hardware_mirror.rviz
```

Terminal 6 — live dashboard:

```bash
ros2 run ras_hardware_mirror live_dashboard --ros-args \
  -p config:=/home/wens/ws_drone_interception/src/ras_hardware_mirror/config/hardware_mirror_dev.yaml \
  -p field:=/home/wens/ws_drone_interception/src/ras_hardware_mirror/config/field.yaml
```


The dashboard command is intentionally a foreground process, so that same
terminal remains occupied until you press `Ctrl-C`. Its Matplotlib window is
non-modal: the keyboard-control terminal, Gazebo, RViz, and other windows remain
clickable. Return to Terminal 4 before sending a flight key.
If the terminal prints `inotify_add_watch ... No space left on device`, that is
an exhausted per-user inotify limit rather than disk space; close an unneeded
editor/IDE instance before restarting the dashboard.

The scientific epoch is published immediately before `RUN`. The target, delay queue, estimator, MPC, and logger reset on that shared transition. A delayed measurement retains its source time in `header.stamp`; receipt/arrival time is logged independently.

## H. Keyboard control and GS1–GS6

Terminal 4 is placed in cbreak mode while the controller runs, so commands are
sent immediately without pressing Enter. SSH terminals do not provide portable
key-release events. Holding a movement key therefore uses the terminal's normal
key repeat; when repeats stop, a 0.30-s watchdog publishes zero velocity. The
terminal settings are restored on `Ctrl-C` or normal shutdown.

| Key | Function |
|---|---|
| `q` | prestream zero velocity, enter offboard, and arm |
| `t` | take off to 2 m, then zero-velocity hover |
| `g` | cancel the current action and land |
| hold `w` / `s` | move up / down |
| hold `a` / `d` | yaw left / right |
| hold ↑ / ↓ | move forward / back relative to current heading |
| hold ← / → | move left / right relative to current heading |

Releasing the movement key stops its repeat stream and commands zero velocity
after the watchdog interval. A movement key during autonomous `RUN` causes
`RUN -> ABORT -> HOLD` and gives the keyboard manual control. Press `g` when
ready to land.

Scenario keys are rejected until PX4 reports `ARMED`:

| Key | Stage | Configuration |
|---|---|---|
| `1` | GS1 | STATIC / DEV0 / M1 |
| `2` | GS2 | HT1 / DEV0 / M1 |
| `3` | GS3 | HT1 / HC0 / M1 |
| `4` | GS4 | HT1 / HC1 / M1 |
| `5` | GS5 M0′ | HT1 / HC1 / M0′, paired seed 6505 |
| `6` | GS5 M1 | HT1 / HC1 / M1, paired seed 6505 |
| `7` | GS6 | eight deterministic non-flight safety checks |

Typical sequence is `q`, wait until Terminal 4 reports armed, optionally `t`,
then press one stage key. Only one scenario should be run per logging session.
The scenario selector atomically updates the virtual target, telemetry
condition, frozen controller mapping, experiment status, and logger before
entering `TAKEOFF`. The stage then climbs to the configured 5 m nominal
experiment altitude, stabilizes for 2 s, and only then publishes `RUN`. The `t`
key remains a separate manual safety takeoff to 2 m; it does not start the
scientific epoch.


Stage	Result
GS0	Gazebo, single PX4, RViz, dashboard, ground truth, and colored markers launched correctly. Manual visual acceptance remains pending.
GS1	Static target captured at 4.440 s; minimum separation 0.999 m; reached DONE.
GS2	HT1/zero delay captured at 8.420 s; minimum separation 0.959 m.
GS3	HT1/HC0 captured; median packet age 51.65 ms.
GS4	HT1/HC1 captured; median packet age 121.27 ms.
GS5 M0′	Captured at 8.660 s; minimum separation 0.992 m.
GS5 M1	Captured at 8.880 s; minimum separation 0.989 m.
GS6	All four safety-transition tests passed.

## I. Run one matched pair

Run the two arms in separate foreground sessions. In the first session press `q`, optionally `t`, then `5`. After safe landing and shutdown, start a clean session and press `q`, optionally `t`, then `6`. Keys 5 and 6 use the same HT1/HC1 geometry and seed 6505 and differ only in M0′ versus M1:

```bash
# Session 1: q, t, 5
bash src/ras_hardware_mirror/scripts/run_mirror_demo.sh

# Session 2: q, t, 6
bash src/ras_hardware_mirror/scripts/run_mirror_demo.sh
```

After both runs reach `DONE`:

```bash
python3 src/ras_hardware_mirror/scripts/analyze_mirror_pair.py \
  results/ras_hardware_mirror/HT1_HC1_M0prime_rep01 \
  results/ras_hardware_mirror/HT1_HC1_M1_rep01 \
  --output results/ras_hardware_mirror/development/HT1_HC1_seed6505_pair
```

The script refuses trajectory/condition/seed mismatches and writes PNG and PDF summaries.

## J. Generate the 24-run manifest

```bash
python3 src/ras_hardware_mirror/scripts/generate_trial_manifest.py
column -s, -t < src/ras_hardware_mirror/manifests/hardware_mirror_24.csv | less -S
```

The manifest has 24 pending rows: HT1/HT2 × HC0/HC1 × M0′/M1 × three repetitions. Paired methods share the same seed. Generating the manifest does not launch anything.

## K. Run the 24 cases manually, one row per session

Automatic campaign execution remains disabled. The terminal keyboard can select
exactly one checksum-validated manifest row with key `8`; it never advances to
the next row and never starts a second trial. The selected row is fixed when the
keyboard node starts.

Keep MicroXRCEAgent and PX4/Gazebo visible in their own foreground terminals.
For row 1, start a fresh experiment-node session in Terminal 3:

```bash
cd /home/wens/ws_drone_interception
source scripts/env.sh
source install/setup.bash
ros2 launch ras_hardware_mirror mirror_nodes.launch.py
```

Then start a fresh keyboard session in Terminal 4:

```bash
cd /home/wens/ws_drone_interception
source scripts/env.sh
source install/setup.bash
ros2 launch ras_hardware_mirror mirror_keyboard.launch.py manifest_row:=1
```

The terminal must print the exact selected row, for example:

```text
SELECTED ROW 01/24: HT1_HC0_M0prime_rep01 seed=11001
```

For that session, use this sequence only:

1. Press `q` once.
2. Wait for `PX4 ARMED` in the keyboard terminal.
3. Check that the printed row/run ID is the intended row.
4. Press `8` once. Do not press a GS key (`1`--`7`).
5. Wait for `DONE`, automatic landing, and PX4 disarm.
6. Confirm that `metadata.json` says `"complete": true` in the exact run
   directory printed by the logger.
7. Press Ctrl-C in the keyboard and experiment-node terminals. Start fresh
   instances for the next row.

Repeat with `manifest_row:=2`, then `3`, through `24`. Never change the CSV to
skip or reorder a trial; select the desired frozen row number explicitly. To
review the numbered mapping before flight:

```bash
awk -F, 'NR==1 {next} {printf "%02d  %-34s seed=%s\n", NR-1, $1, $8}' \
  src/ras_hardware_mirror/manifests/hardware_mirror_24.csv
```

Before each row, refuse to start if its deterministic output directory already
contains a completed `metadata.json`. A miss is a scientific outcome, not a
reason to repeat or tune. Repeat a row only for a separately documented
infrastructure-invalid event. Paired M0-prime/M1 rows already carry the same
seed in the frozen manifest.

The one-command demo also accepts a selected row:

```bash
ros2 launch ras_hardware_mirror mirror_demo.launch.py manifest_row:=1
```

The multi-terminal workflow is preferred because every process remains visible
and independently stoppable.

## L. RViz display

The fixed frame is `map` (East, North, Up). The blue arrow is the X500 common PX4 state, orange sphere is virtual-target truth, purple cube is target estimate, green points/line are the 20-step predicted horizon, black/dark truth path and blue interceptor path are separate `Path` displays, the translucent sphere is capture radius, the red rectangle is hard geofence, the amber rectangle is target operating area, the flat blue cylinder is the initial X500 position, and the text marker reports method/state/separation. Shape and line semantics supplement color. Minimal TF children are `interceptor_base`, `virtual_target`, and `target_estimate` under `map`.

## M. Live dashboard

Panel A is equal-scale top-down ENU truth, estimate, prediction, X500, capture radius, geofence, and target region. Panel B is virtual separation with the capture-radius line. Panel C is current-time target-position error. Panel D compares requested delay to actual source-to-arrival packet age. The header reports method, trajectory, condition, experiment/PX4 state, packet age, separation, error, capture, and safety.

HC0 should settle near 50 ms and HC1 near 120 ms, subject to ROS timer scheduling. Under HC1, M0′ may visibly show more temporal lag/error than M1 if the established mechanism is operating as expected; this rehearsal does not assert that M1 must capture or outperform in any new run. The dashboard subscribes directly to ROS and uses bounded memory; it never polls CSV and never changes controller behavior.

## N. Logs

Each run uses `results/ras_hardware_mirror/<trajectory>_<condition>_<method>_repNN/` and contains:

```text
metadata.json
config_snapshot.yaml
field_snapshot.yaml
steps.csv
telemetry_packets.csv
experiment_events.csv
```

`steps.csv` records truth p/v/a, estimate, PX4/common state, Gazebo truth when the odometry bridge is available, raw/applied command, virtual separation/minimum, capture, PX4/safety state, JSON prediction horizon, and complete-loop latency. `telemetry_packets.csv` records packet ID, source time, independent arrival time, actual/requested age, and drop status. `experiment_events.csv` records state transitions. `metadata.json` is marked complete only after `DONE`. An existing completed deterministic run is never overwritten; an incomplete collision receives a UTC suffix. Rosbag2 is optional and deliberately not required by the demo.

Development screenshots, when captured manually after visual acceptance, belong in `results/ras_hardware_mirror/development/screenshots/`.

The logger prints the absolute directory in the Terminal 3 output when it
starts and again after a scenario key is accepted. Before a scenario is
selected, the temporary directory is named `PENDING_keyboard_<UTC>`; selecting
a stage renames it to the deterministic run name above. From the workspace,
inspect the newest logger directory with:

```bash
find results/ras_hardware_mirror -mindepth 1 -maxdepth 1 -type d \
  -printf '%T@ %p\n' | sort -nr | head
```

## O. Safe shutdown

Normal capture disables pursuit immediately, holds, then lands if `auto_land` is true. Keyboard motion during a run provides a controlled `ABORT -> HOLD` manual takeover; `g` requests landing. PX4 and Gazebo are launched in the foreground by the standard PX4 `make` target—there is no `nohup`, daemon wrapper, or automatic MicroXRCEAgent. Stop experiment nodes, keyboard/RViz/dashboard, then press Ctrl-C in the PX4/Gazebo terminal. If a terminal was killed abruptly, the recovery script searches only for the exact PX4 SITL binary and `gz sim` command before stopping them:

```bash
bash src/ras_hardware_mirror/scripts/stop_mirror_sim.sh
```

Inspect before any manual signal:

```bash
runtime=$(realpath results/ras_hardware_mirror/development/runtime/current)
for file in "$runtime"/*.pid; do printf '%s: ' "$file"; cat "$file"; done
```

Do not use broad commands such as `killall python3`. Stop the manually launched MicroXRCEAgent with Ctrl-C in its own terminal.

## P. Transition to physical RTK and Real Hardware

Later integration replaces `GazeboStateProvider` and the SITL PX4 transport with an `RTKStateProvider` plus the real PX4 transport, after receiver, lever-arm, origin, covariance/quality, field, altitude, and flight-envelope characterization. It preserves the virtual target, telemetry emulator, M0′/M1 adapter, estimator, predictor, MPC, experiment manager, safety supervisor, public topics, RViz, dashboard, logger, and manifest format.

Before final outdoor trials, measured site/RTK characterization must freeze field dimensions, map origin, geofence, target region, flight altitude, capture radius, initial geometry, speed, acceleration, timeout, and GNSS quality criteria in `hardware_mirror_dev.yaml` and `field.yaml`.

## Q. Real Hardware Terminal-by-Terminal Execution

When deploying to physical hardware (e.g. companion computer mounted on an X500 / Pixhawk quadrotor):

### Hardware & Parameter Setup
1. Connect companion computer to Pixhawk via Serial (TELEM2 / USB `/dev/ttyACM0`) or Ethernet.
2. In QGroundControl, set `UXRCE_DDS_CFG` to the companion port, `SER_TEL2_BAUD` to `921600`, and set offboard failsafe `COM_OF_LOSS_T` to `1.0` s.
3. Ensure RTK base station is broadcasting corrections and PX4 EKF2 reports `RTK Fixed` (Fix Type 6).
4. Update `config/field.yaml` and `config/hardware_mirror_dev.yaml` with the outdoor field origin and geofence coordinates.

### Terminal Commands

In each terminal, source the workspace:
```bash
cd /home/wens/ws_drone_interception
source scripts/env.sh
source install/setup.bash
```

- **Terminal 1 — MicroXRCEAgent (Physical Serial / Ethernet Bridge):**
  ```bash
  MicroXRCEAgent serial --dev /dev/ttyACM0 -b 921600
  # Or over Ethernet: MicroXRCEAgent udp4 -p 8888
  ```

- **Terminal 2 — Physical Drone:**
  *(No Gazebo / SITL command needed. The physical drone running PX4 replaces this terminal).*

- **Terminal 3 — Experiment Nodes:**
  ```bash
  ros2 launch ras_hardware_mirror mirror_nodes.launch.py
  # Or with specific arguments:
  # ros2 launch ras_hardware_mirror mirror_nodes.launch.py method:=M1 trajectory:=HT1 condition:=HC1
  ```

- **Terminal 4 — Operator Keyboard Control:**
  ```bash
  ros2 run ras_hardware_mirror keyboard_control
  # Or for a specific manifest trial:
  # ros2 launch ras_hardware_mirror mirror_keyboard.launch.py manifest_row:=1
  ```

- **Terminal 5 (GCS Laptop) — RViz2 Visualizer:**
  ```bash
  rviz2 -d src/ras_hardware_mirror/config/rviz_hardware_mirror.rviz
  ```

- **Terminal 6 (GCS Laptop) — Live Dashboard:**
  ```bash
  ros2 run ras_hardware_mirror live_dashboard --ros-args \
    -p config:=/home/wens/ws_drone_interception/src/ras_hardware_mirror/config/hardware_mirror_dev.yaml \
    -p field:=/home/wens/ws_drone_interception/src/ras_hardware_mirror/config/field.yaml
  ```

### Operator Safety Protocol
1. Press `q` in Terminal 4 to arm and engage Offboard mode.
2. Press `t` to verify stable hover at 2.0 m manual takeoff altitude.
3. Press `1`–`6` (or `8` for loaded manifest row) to execute the scenario.
4. **Manual Abort**: Press `g` to land, touch any motion key (`w/s/a/d/arrows`) to instantly abort pursuit, or switch flight modes on your RC transmitter (Kill Switch / Position mode).

