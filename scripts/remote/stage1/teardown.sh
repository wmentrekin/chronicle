#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-}"
ZONE="${ZONE:-us-east1-c}"
INSTANCE_NAME="${INSTANCE_NAME:-}"
DRY_RUN="${DRY_RUN:-1}"

if [[ -z "${PROJECT_ID}" || -z "${INSTANCE_NAME}" ]]; then
  echo "PROJECT_ID and INSTANCE_NAME are required." >&2
  exit 1
fi

cmd=(
  gcloud compute instances delete "${INSTANCE_NAME}"
  --project "${PROJECT_ID}"
  --zone "${ZONE}"
  --quiet
)

echo "Teardown command:"
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" != "0" ]]; then
  echo "Dry run only. Set DRY_RUN=0 to execute."
  exit 0
fi

"${cmd[@]}"
