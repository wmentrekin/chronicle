#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-}"
ZONE="${ZONE:-us-central1-a}"
INSTANCE_NAME="${INSTANCE_NAME:-}"
SESSION_ID="${SESSION_ID:-}"
LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_DIR:-./outputs}"
WORKER_OUTPUT_DIR="${WORKER_OUTPUT_DIR:-/home/${USER}/chronicle/outputs}"
DRY_RUN="${DRY_RUN:-1}"

if [[ -z "${PROJECT_ID}" || -z "${INSTANCE_NAME}" || -z "${SESSION_ID}" ]]; then
  echo "PROJECT_ID, INSTANCE_NAME, and SESSION_ID are required." >&2
  exit 1
fi

source_dir="${INSTANCE_NAME}:${WORKER_OUTPUT_DIR}/${SESSION_ID}"
target_root="${LOCAL_OUTPUT_DIR%/}"
cmd=(
  gcloud compute scp --recurse
  --project "${PROJECT_ID}"
  --zone "${ZONE}"
  "${source_dir}"
  "${target_root}/"
)

echo "Download command:"
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" != "0" ]]; then
  echo "Dry run only. Set DRY_RUN=0 to execute."
  exit 0
fi

mkdir -p "${target_root}"
"${cmd[@]}"
