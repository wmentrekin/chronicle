#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-}"
ZONE="${ZONE:-us-east1-c}"
INSTANCE_NAME="${INSTANCE_NAME:-}"
SESSION_ID="${SESSION_ID:-}"
LOCAL_SESSION_DIR="${LOCAL_SESSION_DIR:-}"
REMOTE_SESSION_DIR="${REMOTE_SESSION_DIR:-$HOME/chronicle-stage1/sessions}"
DRY_RUN="${DRY_RUN:-1}"

if [[ -z "${PROJECT_ID}" || -z "${INSTANCE_NAME}" || -z "${SESSION_ID}" || -z "${LOCAL_SESSION_DIR}" ]]; then
  echo "PROJECT_ID, INSTANCE_NAME, SESSION_ID, and LOCAL_SESSION_DIR are required." >&2
  exit 1
fi

source_dir="${LOCAL_SESSION_DIR%/}"
target="${INSTANCE_NAME}:${REMOTE_SESSION_DIR}/${SESSION_ID}"
rsync_cmd=(
  rsync -av --delete
  "${source_dir}/"
  "${target}/"
)

ssh_cmd=(
  gcloud compute ssh "${INSTANCE_NAME}"
  --project "${PROJECT_ID}"
  --zone "${ZONE}"
  --command "mkdir -p '${REMOTE_SESSION_DIR}/${SESSION_ID}/audio'"
)

echo "Remote prep command:"
printf ' %q' "${ssh_cmd[@]}"
printf '\n'
echo "Upload command:"
printf ' %q' "${rsync_cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" != "0" ]]; then
  echo "Dry run only. Set DRY_RUN=0 to execute."
  exit 0
fi

"${ssh_cmd[@]}"
"${rsync_cmd[@]}"
