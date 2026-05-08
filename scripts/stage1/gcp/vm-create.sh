#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-}"
ZONE="${ZONE:-us-central1-a}"
INSTANCE_NAME="${INSTANCE_NAME:-chronicle-stage1-${USER:-user}}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-standard-8}"
GPU_TYPE="${GPU_TYPE:-nvidia-l4}"
GPU_COUNT="${GPU_COUNT:-1}"
GPU_ENABLED="${GPU_ENABLED:-0}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-50GB}"
IMAGE_FAMILY="${IMAGE_FAMILY:-ubuntu-2404-lts-amd64}"
IMAGE_PROJECT="${IMAGE_PROJECT:-ubuntu-os-cloud}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-}"
TAGS="${TAGS:-chronicle-stage1}"
LABELS="${LABELS:-chronicle=1,stage=stage1,mode=cloud}"
DRY_RUN="${DRY_RUN:-1}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "PROJECT_ID is required." >&2
  exit 1
fi

cmd=(
  gcloud compute instances create "${INSTANCE_NAME}"
  --project "${PROJECT_ID}"
  --zone "${ZONE}"
  --machine-type "${MACHINE_TYPE}"
  --boot-disk-size "${BOOT_DISK_SIZE}"
  --image-family "${IMAGE_FAMILY}"
  --image-project "${IMAGE_PROJECT}"
  --restart-on-failure
  --tags "${TAGS}"
  --labels "${LABELS}"
)

if [[ "${GPU_ENABLED}" == "1" ]]; then
  cmd+=(--accelerator "type=${GPU_TYPE},count=${GPU_COUNT}")
  cmd+=(--maintenance-policy TERMINATE)
fi

if [[ -n "${SERVICE_ACCOUNT}" ]]; then
  cmd+=(--service-account "${SERVICE_ACCOUNT}")
fi

echo "VM create command:"
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" != "0" ]]; then
  echo "Dry run only. Set DRY_RUN=0 to execute."
  exit 0
fi

"${cmd[@]}"
