# Intel NUC transfer and Codex handoff

## What this bundle contains

This is a minimal source bundle for the ROS 2 hardware-mirror workflow. It
contains the three ROS packages, the frozen controller snapshot, required
configuration, dependency locks, validation tooling, and the detailed Codex
prompt. It deliberately contains no `results/`, `data/`, papers, notebooks,
`build/`, `install/`, or `log/` directory.

## Required destination on the NUC

Install Ubuntu 22.04 LTS on the Intel NUC and create/use the account `zacky`.
The inner `ws_drone_interception` directory must end up exactly here:

```text
/home/zacky/ws_drone_interception
```

Do not place it under `Documents` on the NUC. `Documents` is only the staging
location on the original computer.

## Network transfer from the original computer

Replace `NUC_IP` with the NUC address:

```bash
rsync -aH --info=progress2 \
  /home/zacky/Documents/ws_drone_interception_nuc_bundle_20260825/ws_drone_interception/ \
  zacky@NUC_IP:/home/zacky/ws_drone_interception/
```

The trailing `/` after the source directory is intentional.

## USB transfer alternative

Copy the `ws_drone_interception` directory from the staging bundle to the USB
drive. On the NUC, replace `/media/zacky/USB_NAME` with the actual mount:

```bash
mkdir -p /home/zacky/ws_drone_interception
rsync -aH --info=progress2 \
  /media/zacky/USB_NAME/ws_drone_interception/ \
  /home/zacky/ws_drone_interception/
```

## Verify immediately after transfer

```bash
cd /home/zacky/ws_drone_interception
bash deployment/verify_nuc_bundle.sh
```

Do not continue unless the final line is:

```text
NUC BUNDLE VERIFICATION PASS
```

## Install Codex on the NUC

The current official Codex CLI instructions for macOS/Linux use the standalone
installer:

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh
```

Official documentation:
<https://developers.openai.com/codex/cli>

Open a new terminal if the installer changed the shell path, then:

```bash
cd /home/zacky/ws_drone_interception
codex
```

On first launch, choose `Sign in with ChatGPT` or another sign-in method offered
by Codex. Then submit this instruction:

```text
Use the complete instructions in CODEX_PROMPT_REPLICATE_INTEL_NUC.md to install
and validate this Intel NUC replication environment. Execute every phase and
write deployment/nuc_install_report.md. Do not run an experiment, arm PX4,
enter offboard mode, take off, tune, or modify any frozen input.
```

Review each privileged apt/install command before approving it. The detailed
prompt requires Codex to stop on platform, checksum, revision, or locked-version
mismatches rather than silently substituting dependencies.

## After Codex finishes

Read:

```text
/home/zacky/ws_drone_interception/deployment/nuc_install_report.md
```

Only proceed to the manual 24-row campaign if all validation gates pass. The
one-row-at-a-time keyboard procedure is in
`src/ras_hardware_mirror/README.md`, section K.

