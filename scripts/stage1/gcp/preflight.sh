#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-}"
ZONE="${ZONE:-us-central1-a}"
REQUIRE_BILLING="${REQUIRE_BILLING:-1}"
GPU_ENABLED="${GPU_ENABLED:-0}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "PROJECT_ID is required." >&2
  exit 1
fi

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud is not installed or not on PATH." >&2
  exit 1
fi

echo "Preflight for Chronicle Stage 1 cloud test"
echo "  project: ${PROJECT_ID}"
echo "  zone:    ${ZONE}"
echo

active_account="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null || true)"
configured_project="$(gcloud config get-value project 2>/dev/null || true)"

if [[ -z "${active_account}" ]]; then
  echo "No active gcloud account found. Run: gcloud auth login" >&2
  exit 1
fi

if [[ "${configured_project}" != "${PROJECT_ID}" ]]; then
  echo "Configured gcloud project does not match PROJECT_ID." >&2
  echo "  configured: ${configured_project:-<none>}" >&2
  echo "  expected:   ${PROJECT_ID}" >&2
  echo "Run: gcloud config set project ${PROJECT_ID}" >&2
  exit 1
fi

echo "Active account: ${active_account}"
echo "Configured project matches."

if gcloud services list --enabled --project "${PROJECT_ID}" --format='value(config.name)' \
  | grep -qx 'compute.googleapis.com'; then
  echo "Compute Engine API: enabled"
else
  echo "Compute Engine API: NOT enabled"
  echo "Enable it with:"
  echo "  gcloud services enable compute.googleapis.com --project ${PROJECT_ID}"
fi

billing_enabled="$(gcloud beta billing projects describe "${PROJECT_ID}" --format='value(billingEnabled)' 2>/dev/null || true)"
case "${billing_enabled}" in
  True|true)
    echo "Billing status: enabled"
    ;;
  False|false|"")
    echo "Billing status: disabled or unavailable"
    if [[ "${REQUIRE_BILLING}" == "1" && "${GPU_ENABLED}" == "1" ]]; then
      echo "GPU VMs cannot be created while the project is in Free Trial-only status."
      echo "You must activate the billing account before the L4/T4 test will run."
      exit 2
    fi
    ;;
esac

if [[ "${GPU_ENABLED}" == "1" ]] && gcloud compute accelerator-types list --zones "${ZONE}" --project "${PROJECT_ID}" \
  --format='value(name)' 2>/dev/null | grep -qx 'nvidia-l4'; then
  echo "Zone check: nvidia-l4 appears available in ${ZONE}"
elif [[ "${GPU_ENABLED}" == "1" ]]; then
  echo "Zone check: nvidia-l4 availability not confirmed in ${ZONE}"
  echo "Check available accelerator types with:"
  echo "  gcloud compute accelerator-types list --zones ${ZONE} --project ${PROJECT_ID}"
else
  echo "Zone check: skipped for CPU-first flow"
fi

echo
echo "Preflight complete."
