#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-}"
ZONE="${ZONE:-us-east1-c}"
INSTANCE_NAME="${INSTANCE_NAME:-}"
SESSION_ID="${SESSION_ID:-}"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-$HOME/chronicle}"
REMOTE_SESSION_DIR="${REMOTE_SESSION_DIR:-$HOME/chronicle-stage1/sessions}"
MODEL_NAME="${MODEL_NAME:-nvidia/parakeet-ctc-0.6b}"
DRY_RUN="${DRY_RUN:-1}"

if [[ -z "${PROJECT_ID}" || -z "${INSTANCE_NAME}" || -z "${SESSION_ID}" ]]; then
  echo "PROJECT_ID, INSTANCE_NAME, and SESSION_ID are required." >&2
  exit 1
fi

remote_command=$(
  cat <<EOF
set -euo pipefail
cd '${REMOTE_REPO_DIR}'
source .venv/bin/activate
export MODEL_NAME='${MODEL_NAME}'
chronicle transcribe '${SESSION_ID}'
EOF
)

cmd=(
  gcloud compute ssh "${INSTANCE_NAME}"
  --project "${PROJECT_ID}"
  --zone "${ZONE}"
  --command "${remote_command}"
)

echo "Run command:"
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" != "0" ]]; then
  echo "Dry run only. Set DRY_RUN=0 to execute."
  exit 0
fi

"${cmd[@]}"
