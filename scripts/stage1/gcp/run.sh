#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-}"
ZONE="${ZONE:-us-central1-a}"
INSTANCE_NAME="${INSTANCE_NAME:-}"
SESSION_ID="${SESSION_ID:-}"
WORKER_REPO_DIR="${WORKER_REPO_DIR:-/home/${USER}/chronicle}"
MODEL_NAME="${MODEL_NAME:-nvidia/parakeet-ctc-0.6b}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
DRY_RUN="${DRY_RUN:-1}"

if [[ -z "${PROJECT_ID}" || -z "${INSTANCE_NAME}" || -z "${SESSION_ID}" ]]; then
  echo "PROJECT_ID, INSTANCE_NAME, and SESSION_ID are required." >&2
  exit 1
fi

worker_command=$(
  cat <<EOF
set -euo pipefail
cd '${WORKER_REPO_DIR}'
export MODEL_NAME='${MODEL_NAME}'
export PATH="\$HOME/.local/bin:\$PATH"
uv run --python '${PYTHON_VERSION}' chronicle transcribe '${SESSION_ID}' --local-worker
EOF
)

cmd=(
  gcloud compute ssh "${INSTANCE_NAME}"
  --project "${PROJECT_ID}"
  --zone "${ZONE}"
  --command "${worker_command}"
)

echo "Run command:"
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" != "0" ]]; then
  echo "Dry run only. Set DRY_RUN=0 to execute."
  exit 0
fi

"${cmd[@]}"
