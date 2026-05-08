# Stage 1 Cloud Runbook

This runbook documents the first verified cloud Stage 1 path on the `remote` branch.

## Proven path

- Provider: Google Cloud
- Compute shape: CPU VM
- Tested machine: `e2-standard-8`
- Tested zone: `us-central1-a`
- Model: `nvidia/parakeet-ctc-0.6b`
- Execution mode: cloud VM, repo clone on worker, inputs copied into the Chronicle repo layout, outputs copied back into the local repo

## What was validated

- cloud VM creation
- repo bootstrap on Ubuntu 24
- `uv` install plus Python 3.11 provisioning
- Stage 1 dependency sync
- Parakeet model fetch
- session upload into `inputs/sessions/<session_id>/`
- participant metadata upload into `inputs/global/participants.yaml`
- `chronicle transcribe <session_id>` execution on the worker
- artifact download back into local `outputs/`
- VM teardown

## Notes

- GPU-backed GCP tests were blocked by repeated regional stockout, not by Chronicle or account configuration.
- The verified path is therefore CPU-first for now.
- The cloud run still preserves Chronicle's filesystem contracts and stage boundaries.

## Verified result shape

- A full private-session cloud validation run completed successfully.
- The resulting run metadata used the standard Stage 1 run record shape under `outputs/<session_id>/runs/`.

## Operator wrappers

- `scripts/stage1/gcp/preflight.sh`
- `scripts/stage1/gcp/vm-create.sh`
- `scripts/stage1/gcp/bootstrap.sh`
- `scripts/stage1/gcp/upload.sh`
- `scripts/stage1/gcp/run.sh`
- `scripts/stage1/gcp/download-results.sh`
- `scripts/stage1/gcp/teardown.sh`

## Python-side orchestration

- `src/chronicle/stage1/orchestration.py` owns the durable Python-side Stage 1 orchestration layer.
- `chronicle transcribe-plan` renders a full command plan from Chronicle config in `src`.
- `chronicle transcribe-command` renders one shell-safe command for a specific orchestration step.
- The shell wrappers are not the source of truth for Stage 1 behavior; they are provider-specific operator helpers.
- `chronicle transcribe` now uses the cloud Stage 1 orchestration path by default, while local Parakeet execution remains an internal worker mode used on the VM.
