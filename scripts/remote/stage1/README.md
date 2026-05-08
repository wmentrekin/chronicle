# Stage 1 Remote Scaffold

This directory contains conservative, repo-side shell templates for a Google Cloud-based Stage 1 transcription workflow.

Defaults:

- `ZONE=us-east1-c`
- `MACHINE_TYPE=g2-standard-4`
- `MODEL_NAME=nvidia/parakeet-ctc-0.6b`
- dry-run mode is enabled unless `DRY_RUN=0`

The scripts are intentionally non-opinionated placeholders. They print the commands they would run and only execute when you opt in with the documented env vars.

Suggested flow:

1. Review and customize `env` values in each script call.
2. Run `vm-create.sh` to inspect the VM creation command.
3. Run `bootstrap.sh` on the VM or via startup-script wiring.
4. Use `upload.sh` to stage one session's inputs.
5. Run `run.sh` to invoke Stage 1 transcription on the VM.
6. Use `download-results.sh` to pull artifacts back.
7. Run `teardown.sh` when finished.

Notes:

- No cloud resources are created by this repository scaffolding.
- The scripts assume you will supply your own `PROJECT_ID`, `INSTANCE_NAME`, and local session path.
- These templates are intentionally narrow and are not wired into `chronicle` yet.
