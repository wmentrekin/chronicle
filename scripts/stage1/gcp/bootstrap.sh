#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-nvidia/parakeet-ctc-0.6b}"
CHRONICLE_REPO_DIR="${CHRONICLE_REPO_DIR:-$HOME/chronicle}"
SESSION_DIR="${SESSION_DIR:-$HOME/chronicle-stage1}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
DRY_RUN="${DRY_RUN:-1}"

cat <<EOF
Bootstrap template for a fresh Stage 1 VM.

Assumptions:
- Chronicle repo will be available at: ${CHRONICLE_REPO_DIR}
- Session working directory: ${SESSION_DIR}
- Stage 1 model: ${MODEL_NAME}

This script is intentionally a placeholder. Add your own OS/package/model setup here.
EOF

commands=(
  "mkdir -p '${SESSION_DIR}'"
  "curl -LsSf https://astral.sh/uv/install.sh | sh"
  "export PATH=\"\$HOME/.local/bin:\$PATH\""
  "uv python install '${PYTHON_VERSION}'"
  "cd '${CHRONICLE_REPO_DIR}' && uv sync --python '${PYTHON_VERSION}' --group dev --group stage1-parakeet"
  "cd '${CHRONICLE_REPO_DIR}' && uv run --python '${PYTHON_VERSION}' chronicle models fetch parakeet"
)

printf 'Bootstrap commands:\n'
for command in "${commands[@]}"; do
  printf '  %s\n' "$command"
done

if [[ "${DRY_RUN}" != "0" ]]; then
  echo "Dry run only. Set DRY_RUN=0 to execute the template commands manually."
  exit 0
fi

mkdir -p "${SESSION_DIR}"
if [[ -d "${CHRONICLE_REPO_DIR}" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  uv python install "${PYTHON_VERSION}"
  (cd "${CHRONICLE_REPO_DIR}" && uv sync --python "${PYTHON_VERSION}" --group dev --group stage1-parakeet)
  (cd "${CHRONICLE_REPO_DIR}" && uv run --python "${PYTHON_VERSION}" chronicle models fetch parakeet)
else
  echo "CHRONICLE_REPO_DIR does not exist: ${CHRONICLE_REPO_DIR}" >&2
  exit 1
fi
