# Codex Prompt — Reproduce the RAS Hardware-Mirror Workspace on an Intel NUC

## 1. Role and objective

You are operating on a new Intel/ASUS NUC. Reproduce the validated ROS 2
hardware-mirror development environment from the supplied minimal workspace at:

```text
/home/zacky/ws_drone_interception
```

The required result is a native x86_64 build of the copied ROS packages and
their locked external dependencies, ready for the human operator to run the
24-row PX4/Gazebo campaign manually, one trial per foreground session.

This task is installation and non-flight validation only. Do not execute a
scientific trial, arm PX4, enter offboard mode, take off, tune a controller,
alter a frozen input, or claim new experimental evidence.

## 2. Non-negotiable constraints

1. Require Ubuntu 22.04 LTS, `x86_64`, Python 3.10, ROS 2 Humble, and the Linux
   account `zacky`. Stop and report if any of these differ. Do not perform an OS
   release upgrade.
2. Keep these paths exact:

   ```text
   /home/zacky/ws_drone_interception
   /home/zacky/acados
   /home/zacky/acados_env
   /home/zacky/Micro-XRCE-DDS-Agent
   /home/zacky/PX4-version1.15.2/PX4-Autopilot
   ```

3. Read `deployment/nuc-source-lock.yaml`, `deployment/apt-packages.lock.txt`,
   `requirements.txt`, `requirements-px4.txt`, and this entire prompt before
   making changes.
4. Never copy `build/`, `install/`, `log/`, ACADOS binaries, PX4 binaries, or
   shared libraries from another computer. Build all of them natively on the
   NUC.
5. Do not download a different controller repository. The exact controller
   dependency is already vendored at
   `vendor/drone-interception-comparison-v3`.
6. Do not edit any file under that vendor directory, the 24-row CSV manifest,
   `configs/dapcs_mpc_v1/imm.yaml`, or the controller YAML.
7. Do not install ROS packages from PyPI. ROS, `rclpy`, message types, RViz, and
   `ros_gz` must come from official Ubuntu/ROS/Gazebo apt repositories.
8. Do not use Docker, Conda, Snap ROS, `nohup`, systemd user services, tmux
   autostart, or hidden background processes.
9. MicroXRCEAgent, PX4, Gazebo, experiment nodes, and keyboard control must
   remain manual foreground commands. Do not start them during installation.
10. Do not erase an existing directory. If a required target path already
    exists and does not match the locked revision, stop and report the path,
    revision, and dirty status.
11. Record every installed version and every deviation in
    `deployment/nuc_install_report.md`.

## 3. Authoritative sources

Use only official upstream documentation/repositories when a repository must be
configured:

- ROS 2 Humble Ubuntu installation:
  <https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html>
- ROS 2 Humble supported platforms:
  <https://docs.ros.org/en/humble/Releases/Release-Humble-Hawksbill.html>
- Gazebo Harmonic binary installation:
  <https://gazebosim.org/docs/harmonic/install_ubuntu/>
- ACADOS installation:
  <https://docs.acados.org/installation/index.html>
- PX4 v1.15 ROS 2 guide:
  <https://docs.px4.io/v1.15/en/ros2/user_guide>

Do not replace locked source revisions with a newer release merely because one
exists.

## 4. Phase A — preflight and bundle integrity

Run these read-only checks first:

```bash
id -un
uname -m
lsb_release -ds
python3 --version
test -f /etc/os-release && . /etc/os-release && printf '%s %s\n' "$ID" "$VERSION_ID"
```

Required answers are `zacky`, `x86_64`, Ubuntu `22.04`, and Python `3.10.x`.
Stop on a mismatch.

Then verify the transferred source without changing it:

```bash
cd /home/zacky/ws_drone_interception
bash deployment/verify_nuc_bundle.sh
git -C src/px4_msgs rev-parse HEAD
git -C src/px4_msgs status --short --branch
```

The bundle validator must end with `NUC BUNDLE VERIFICATION PASS`, and
`px4_msgs` must resolve to:

```text
a1045ec4feb6d709bdecaf3895f1d5b43a5dabb8
```

Do not continue if a frozen checksum fails. Do not “repair” a checksum by
editing either the input or expected digest.

Confirm that the transfer contains none of the excluded large/generated roots:

```bash
for path in results data build install log; do
  test ! -e "/home/zacky/ws_drone_interception/$path" || exit 1
done
```

Start `deployment/nuc_install_report.md` with the preflight output, date,
hostname, kernel, CPU model, RAM, disk, and bundle-verification result.

## 5. Phase B — configure official apt repositories

If ROS 2 Humble is not already configured, follow the current official ROS 2
Humble Ubuntu Debian instructions linked above. Enable Ubuntu `universe`, use
the official ROS apt-source/key mechanism, and run `sudo apt update`.

If Gazebo Harmonic packages are unavailable, configure the official Gazebo
packages repository using the Harmonic instructions linked above. Do not add a
third-party mirror.

Before installation, inspect candidates for representative packages:

```bash
apt-cache policy ros-humble-ros-base ros-humble-rclpy \
  ros-humble-ros-gzharmonic-sim gz-harmonic python3-gz-transport13
```

Install the exact apt specifications recorded on the reference machine:

```bash
cd /home/zacky/ws_drone_interception
mapfile -t NUC_APT_SPECS < <(sed -e '/^[[:space:]]*#/d' -e '/^[[:space:]]*$/d' \
  deployment/apt-packages.lock.txt)
sudo apt-get install -y "${NUC_APT_SPECS[@]}"
```

If apt reports that an exact version is unavailable, do not silently install a
different version. Record the candidate and locked version in the report and
stop for user approval. Exact apt reproducibility cannot be claimed without
the locked version or an approved, documented deviation.

Initialize rosdep only if it is not already initialized:

```bash
sudo rosdep init
rosdep update --rosdistro humble
```

Treat “sources list file already exists” from `rosdep init` as an already
initialized state, not a reason to delete anything.

## 6. Phase C — clone and prepare PX4 v1.15.2

The PX4 tree is external to the transferred workspace. Clone it at the exact
locked location and revision:

```bash
mkdir -p /home/zacky/PX4-version1.15.2
git clone --recursive --branch v1.15.2 \
  https://github.com/PX4/PX4-Autopilot.git \
  /home/zacky/PX4-version1.15.2/PX4-Autopilot
cd /home/zacky/PX4-version1.15.2/PX4-Autopilot
git checkout 4817c0618a1286846116e90c6eb8919efaa013cf
git submodule sync --recursive
git submodule update --init --recursive
```

Verify the parent and Gazebo-model submodule revisions:

```bash
test "$(git rev-parse HEAD)" = 4817c0618a1286846116e90c6eb8919efaa013cf
test "$(git -C Tools/simulation/gz rev-parse HEAD)" = d754381a1cecdd7f17050acd72bf5bf1327bced6
git status --short --branch
```

The PX4 checkout must be clean. The old reference machine had an optional UWB
visual-only model edit; it is deliberately excluded and is not required by the
ROS workflow.

Install PX4 host build prerequisites without NuttX or PX4's older simulator
package selection:

```bash
bash Tools/setup/ubuntu.sh --no-nuttx --no-sim-tools
```

That upstream script contains version ranges. Immediately restore the recorded
PX4 host-tool and experiment runtime pins:

```bash
cd /home/zacky/ws_drone_interception
/usr/bin/python3 -m pip install --user -r requirements-px4.txt
/usr/bin/python3 -m pip install --user -r requirements.txt
```

Build SITL without launching a simulator:

```bash
cd /home/zacky/PX4-version1.15.2/PX4-Autopilot
make px4_sitl_default
test -x build/px4_sitl_default/bin/px4
```

Do not run `make px4_sitl gz_x500` during installation because it starts PX4
and Gazebo interactively.

## 7. Phase D — build ACADOS v0.6.0

Clone recursively at the exact locked revision:

```bash
git clone --recursive https://github.com/acados/acados.git /home/zacky/acados
cd /home/zacky/acados
git checkout 503364817c872d474ab5bed219c26760ac267769
git submodule sync --recursive
git submodule update --init --recursive
test "$(git rev-parse HEAD)" = 503364817c872d474ab5bed219c26760ac267769
```

Configure and build with the reference x86 solver options:

```bash
cmake -S /home/zacky/acados -B /home/zacky/acados/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/home/zacky/acados \
  -DACADOS_WITH_QPOASES=ON \
  -DACADOS_WITH_OPENMP=OFF \
  -DBLASFEO_TARGET=X64_AUTOMATIC \
  -DHPIPM_TARGET=GENERIC
cmake --build /home/zacky/acados/build --parallel "$(nproc)"
cmake --install /home/zacky/acados/build
```

Create the ACADOS virtual environment at the same location used by the
reference machine. Do not install ROS into it:

```bash
/usr/bin/python3 -m venv /home/zacky/acados_env
source /home/zacky/acados_env/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-deps -e /home/zacky/acados/interfaces/acados_template
deactivate
```

The runtime pins remain in the `/usr/bin/python3` user site because ROS 2
ament-python executables have a `/usr/bin/python3` shebang. `scripts/env.sh`
adds both the ACADOS template source and user site to `PYTHONPATH`.

Verify native ACADOS artifacts:

```bash
test -f /home/zacky/acados/lib/libacados.so
grep -E '^(BLASFEO_TARGET|HPIPM_TARGET|ACADOS_WITH_QPOASES|ACADOS_WITH_OPENMP):' \
  /home/zacky/acados/build/CMakeCache.txt
```

## 8. Phase E — build Micro XRCE-DDS Agent v2.4.3

Clone and build natively:

```bash
git clone --recursive --branch v2.4.3 \
  https://github.com/eProsima/Micro-XRCE-DDS-Agent.git \
  /home/zacky/Micro-XRCE-DDS-Agent
cd /home/zacky/Micro-XRCE-DDS-Agent
git checkout 73622810d984349b80bbac0ef55fc0b694d62222
git submodule sync --recursive
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel "$(nproc)"
sudo cmake --install build
sudo ldconfig
```

Verify the revision and executable, but do not start the agent:

```bash
test "$(git rev-parse HEAD)" = 73622810d984349b80bbac0ef55fc0b694d62222
command -v MicroXRCEAgent
```

## 9. Phase F — build the ROS 2 workspace

Return to a fresh shell state, then source the supplied environment:

```bash
cd /home/zacky/ws_drone_interception
source scripts/env.sh
```

Confirm the environment resolves only local/copied controller sources:

```bash
test "$WS_DRONE_INTERCEPTION" = /home/zacky/ws_drone_interception
test "$DRONE_INTERCEPTION_V3" = /home/zacky/ws_drone_interception/vendor/drone-interception-comparison-v3
test "$ACADOS_SOURCE_DIR" = /home/zacky/acados
test "$PX4_DIR" = /home/zacky/PX4-version1.15.2/PX4-Autopilot
```

Run rosdep in simulation mode only, excluding the `ament_python` build type and
`ament_pytest` test helper (which Humble's rosdep index does not resolve as
system keys) and the two Python dependencies supplied by the locked user-site
requirements. This is a dependency audit, not authorization to install an
unrecorded package:

```bash
rosdep install --simulate --from-paths src --ignore-src --rosdistro humble \
  --skip-keys "ament_python ament_pytest python3-pandas python3-matplotlib"
```

If this proposes any package not already present in
`deployment/apt-packages.lock.txt`, stop and record it instead of installing it
silently.

The pinned user-site setuptools is newer than the Humble colcon path expects.
Build with the same isolation used by the reference machine:

```bash
cd /home/zacky/ws_drone_interception
source scripts/env.sh
NUC_CLEAN_PYTHONPATH=$(/usr/bin/python3 - <<'PY'
import os
print(':'.join(
    p for p in os.environ.get('PYTHONPATH', '').split(':')
    if p and '/home/zacky/.local/lib/python3.10/site-packages' not in p
))
PY
)
PYTHONNOUSERSITE=1 PYTHONPATH="$NUC_CLEAN_PYTHONPATH" colcon build \
  --packages-up-to ras_hardware_mirror
source install/setup.bash
```

Do not reuse or import any `build/` or `install/` directory from the original
computer.

## 10. Phase G — non-flight validation

Run environment validation and tests without launching ROS nodes:

```bash
cd /home/zacky/ws_drone_interception
source scripts/env.sh
source install/setup.bash
bash src/ras_hardware_mirror/scripts/validate_environment.sh
colcon test --packages-select ras_hardware_mirror \
  --python-testing pytest --event-handlers console_direct+
colcon test-result --test-result-base build/ras_hardware_mirror --verbose
```

Required results:

- `ENVIRONMENT VALIDATION PASS`
- 20 tests, zero errors, zero failures

Verify the installed manifest and the exact 24 operator selections:

```bash
/usr/bin/python3 - <<'PY'
from ras_hardware_mirror.manifest_utils import default_manifest, load_campaign_scenarios
rows = load_campaign_scenarios()
assert len(rows) == 24
assert rows[1]['run_id'] == 'HT1_HC0_M0prime_rep01'
assert rows[24]['run_id'] == 'HT2_HC1_M1_rep03'
for index in range(1, 25, 2):
    left, right = rows[index], rows[index + 1]
    assert left['seed'] == right['seed']
    assert left['trajectory'] == right['trajectory']
    assert left['condition'] == right['condition']
print('24-row installed manifest PASS:', default_manifest())
PY
```

Verify launch arguments without starting anything:

```bash
ros2 launch ras_hardware_mirror mirror_gazebo.launch.py --show-args
ros2 launch ras_hardware_mirror mirror_nodes.launch.py --show-args
ros2 launch ras_hardware_mirror mirror_keyboard.launch.py --show-args
```

The keyboard launch must expose `manifest_row` with default `0`.

Instantiate both frozen controller arms as a build/import test only. Use a
temporary working directory so generated ACADOS code is not placed in the
workspace:

```bash
NUC_SOLVER_CHECK=$(mktemp -d)
cd "$NUC_SOLVER_CHECK"
/usr/bin/python3 - <<'PY'
import os
from pathlib import Path
from m0prime_confirmatory.controller import ConfirmatoryControllerAdapter

root = Path(os.environ['WS_DRONE_INTERCEPTION'])
config = root / 'vendor/drone-interception-comparison-v3/configs/q2_revision_pilot.yaml'
for method in ('A0prime_CA_arrival', 'mpc_dca_tracking'):
    adapter = ConfirmatoryControllerAdapter(method, config, 11001)
    assert adapter.method == method
    print('controller construction PASS', method)
PY
```

Do not delete the temporary directory automatically. Record its path; the user
may remove it after inspecting the generated build log.

Re-run source integrity after all builds. This mode permits locally generated
`build/`, `install/`, `log/`, caches, and native libraries while still checking
every inventoried transferred source file:

```bash
cd /home/zacky/ws_drone_interception
bash deployment/verify_nuc_bundle.sh --allow-generated
```

## 11. Phase H — final audit and handoff

Confirm that no experiment-related process was started:

```bash
ps -eo pid,ppid,stat,cmd | grep -E '[M]icroXRCEAgent|[p]x4|[g]z sim|[r]as_hardware_mirror'
```

An empty result is expected. Do not kill unrelated processes.

Complete `deployment/nuc_install_report.md` with:

1. host/OS/architecture;
2. apt repository provenance and exact installed versions;
3. Python package versions from `/usr/bin/python3`;
4. PX4, PX4 Gazebo submodule, `px4_msgs`, ACADOS, and Micro XRCE-DDS commits;
5. ACADOS CMake target/options;
6. bundle checksum result before and after build;
7. colcon build result;
8. environment-validator result;
9. test count and result;
10. controller-construction result and temporary directory;
11. every deviation or unresolved warning;
12. explicit statement: `No experiment, arming, takeoff, or tuning was performed.`

Installation is complete only when every required gate passes. If a gate
fails, diagnose the installation problem, but do not modify a frozen scientific
input or run a trial as a workaround.

## 12. Operator commands after installation

Do not execute these as part of this prompt. Present them to the human after a
successful handoff.

Terminal 1:

```bash
MicroXRCEAgent udp4 -p 8888
```

Terminal 2:

```bash
cd /home/zacky/ws_drone_interception
source scripts/env.sh
source install/setup.bash
ros2 launch ras_hardware_mirror mirror_gazebo.launch.py gui:=--gui
```

Terminal 3, restarted for every manifest row:

```bash
cd /home/zacky/ws_drone_interception
source scripts/env.sh
source install/setup.bash
ros2 launch ras_hardware_mirror mirror_nodes.launch.py
```

Terminal 4, where `N` is exactly one integer from 1 through 24:

```bash
cd /home/zacky/ws_drone_interception
source scripts/env.sh
source install/setup.bash
ros2 launch ras_hardware_mirror mirror_keyboard.launch.py manifest_row:=N
```

The human presses `q`, waits for `PX4 ARMED`, verifies the displayed row, then
presses `8` exactly once. The software never advances to another manifest row.
After `DONE`, landing, and disarm, the human verifies `metadata.json`, stops
Terminals 3 and 4, and starts fresh sessions for the next row.

The complete one-row-at-a-time protocol is in
`src/ras_hardware_mirror/README.md`, section K.

## 13. Scientific and hardware scope warning

The copied `ras_hardware_mirror` package is a one-X500 PX4 SITL/Gazebo
rehearsal. `GazeboStateProvider` is implemented; the real `RTKStateProvider` is
still a stub. This installation must not be described as physical-flight or
RTK validation, and the current launch file must not be used to command a real
aircraft.
