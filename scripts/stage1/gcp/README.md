# Stage 1 GCP Operator Scripts

This directory contains conservative GCP shell wrappers for the default cloud-based Stage 1 workflow.

Defaults:

- `ZONE=us-central1-a`
- `MACHINE_TYPE=e2-standard-8`
- `MODEL_NAME=nvidia/parakeet-ctc-0.6b`
- `BOOT_DISK_SIZE=50GB`
- dry-run mode is enabled unless `DRY_RUN=0`

The scripts are intentionally conservative templates. They print the commands they would run and only execute when you opt in with `DRY_RUN=0`.

Important billing note:

- Google Cloud Free Trial credits are not enough by themselves for GPU-backed workers.
- Google documents that Free Trial accounts cannot add GPUs to VM instances.
- CPU workers do not need the GPU quota flow.

Suggested flow:

1. Run `preflight.sh` first.
2. Review and customize `env` values in each script call.
3. Run `vm-create.sh` to inspect the VM creation command.
4. Run `bootstrap.sh` on the VM or via startup-script wiring.
5. Use `upload.sh` to stage one session's inputs.
6. Run `run.sh` to invoke Stage 1 on the worker.
7. Use `download-results.sh` to pull artifacts back.
8. Run `teardown.sh` immediately when finished.

Notes:

- No cloud resources are created by this repository scaffolding.
- The scripts assume you will supply your own `PROJECT_ID`, `INSTANCE_NAME`, `SESSION_ID`, and local session path.
- Uploads target the Chronicle repo layout under `inputs/sessions/<session_id>/`.
- The scripts are intentionally narrow wrappers around the Python-side planner in `src/chronicle/stage1/orchestration.py`.
- `preflight.sh` exits non-zero if billing is not activated and the target flow requires a GPU worker.

## Minimal operator flow

Dry-run the workflow:

```bash
PROJECT_ID=<project-id> \
bash scripts/stage1/preflight.sh || true

PROJECT_ID=<project-id> \
INSTANCE_NAME=<instance-name> \
bash scripts/stage1/vm-create.sh
```

The default CPU-first flow should use:

- `ZONE=us-central1-a`
- `MACHINE_TYPE=e2-standard-8`
- `GPU_ENABLED=0`
- `BOOT_DISK_SIZE=50GB`
- `SESSION_ID=<session_id>`

If you want to try a GPU-backed worker:

- `GPU_ENABLED=1`
- `ZONE=<gpu-zone>`
- `MACHINE_TYPE=<gpu-machine-type>`
- `GPU_TYPE=<gpu-type>`
