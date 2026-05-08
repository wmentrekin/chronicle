#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-}"
ZONE="${ZONE:-us-central1-a}"
INSTANCE_NAME="${INSTANCE_NAME:-}"
SESSION_ID="${SESSION_ID:-}"
LOCAL_SESSION_DIR="${LOCAL_SESSION_DIR:-}"
WORKER_REPO_DIR="${WORKER_REPO_DIR:-/home/${USER}/chronicle}"
LOCAL_PARTICIPANTS_FILE="${LOCAL_PARTICIPANTS_FILE:-}"
DRY_RUN="${DRY_RUN:-1}"

if [[ -z "${PROJECT_ID}" || -z "${INSTANCE_NAME}" || -z "${SESSION_ID}" || -z "${LOCAL_SESSION_DIR}" || -z "${LOCAL_PARTICIPANTS_FILE}" ]]; then
  echo "PROJECT_ID, INSTANCE_NAME, SESSION_ID, LOCAL_SESSION_DIR, and LOCAL_PARTICIPANTS_FILE are required." >&2
  exit 1
fi

source_dir="${LOCAL_SESSION_DIR%/}"
worker_session_root="${WORKER_REPO_DIR}/inputs/sessions"
worker_global_root="${WORKER_REPO_DIR}/inputs/global"

ssh_cmd=(
  gcloud compute ssh "${INSTANCE_NAME}"
  --project "${PROJECT_ID}"
  --zone "${ZONE}"
  --command "mkdir -p '${worker_session_root}' '${worker_global_root}'"
)

session_copy_cmd=(
  gcloud compute scp --recurse
  --project "${PROJECT_ID}"
  --zone "${ZONE}"
  "${source_dir}"
  "${INSTANCE_NAME}:${worker_session_root}/"
)

participants_copy_cmd=(
  gcloud compute scp
  --project "${PROJECT_ID}"
  --zone "${ZONE}"
  "${LOCAL_PARTICIPANTS_FILE}"
  "${INSTANCE_NAME}:${worker_global_root}/participants.yaml"
)

echo "Worker prep command:"
printf ' %q' "${ssh_cmd[@]}"
printf '\n'
echo "Session upload command:"
printf ' %q' "${session_copy_cmd[@]}"
printf '\n'
echo "Participants upload command:"
printf ' %q' "${participants_copy_cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" != "0" ]]; then
  echo "Dry run only. Set DRY_RUN=0 to execute."
  exit 0
fi

"${ssh_cmd[@]}"
"${session_copy_cmd[@]}"
"${participants_copy_cmd[@]}"
