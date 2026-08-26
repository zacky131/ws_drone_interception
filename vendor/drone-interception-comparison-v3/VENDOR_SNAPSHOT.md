# Frozen controller snapshot

This directory is the self-contained source/configuration dependency used by
`drone_interception_px4` and `ras_hardware_mirror` for M0-prime and M1. It was
copied from the previously external, non-Git directory:

`/media/zacky/ZacksSSD/RESEARCH/DroneInterception/drone-interception-comparison-v3`

Only source code, scripts, configuration, the original requirements file, and
the original README are included. Results, caches, generated solvers, and
compiled files are deliberately excluded. Do not tune or rewrite this snapshot
during deployment. Its tree digest and the frozen YAML digest are recorded in
`deployment/source-lock.yaml` at the workspace root.

