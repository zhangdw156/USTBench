#!/usr/bin/env bash
set -euo pipefail

# Sync local USTBench evaluation results to a Hugging Face bucket.
#
# Default source:
#   results/
#
# Default destination:
#   hf://buckets/zhangdw/leo-benchmark/USTBench/results
#
# Examples:
#   bash scripts/sync_results_to_hf.sh --dry-run
#   bash scripts/sync_results_to_hf.sh
#   bash scripts/sync_results_to_hf.sh --delete
#   bash scripts/sync_results_to_hf.sh -- --include "*.json"
#
# Authentication is handled by the hf CLI. Use `hf auth login` or set HF_TOKEN.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/results"
DEST_URI="hf://buckets/zhangdw/leo-benchmark/USTBench/results"
DRY_RUN=0
DELETE=0
ALLOW_EMPTY=0
EXTRA_ARGS=()

abs_path() {
  python3 - "$1" <<'PYABS'
import os
import sys
print(os.path.abspath(sys.argv[1]))
PYABS
}

usage() {
  cat <<EOF
Usage: $0 [options] [-- extra hf sync args]

Sync local USTBench evaluation results to a Hugging Face bucket using:
  uvx hf buckets sync

Options:
  --results-dir DIR     Local results directory.
                        Default: ${RESULTS_DIR}
  --dest URI            Destination bucket URI.
                        Default: ${DEST_URI}
  --dry-run             Print the sync plan without uploading.
  --delete              Delete destination files that are not present locally.
                        Disabled by default.
  --allow-empty         Allow syncing when the local results directory has no files.
  -h, --help            Show this help.

Examples:
  bash scripts/sync_results_to_hf.sh --dry-run
  bash scripts/sync_results_to_hf.sh
  bash scripts/sync_results_to_hf.sh --delete
  bash scripts/sync_results_to_hf.sh -- --include "*.json"
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --results-dir)
      RESULTS_DIR="$(abs_path "${2:?Missing value for --results-dir}")"
      shift 2
      ;;
    --dest)
      DEST_URI="${2:?Missing value for --dest}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --delete)
      DELETE=1
      shift
      ;;
    --allow-empty)
      ALLOW_EMPTY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v uvx >/dev/null 2>&1; then
  echo "ERROR: uvx is required but was not found in PATH." >&2
  exit 1
fi

if [[ ! -d "${RESULTS_DIR}" ]]; then
  cat >&2 <<EOF
ERROR: results directory does not exist:
  ${RESULTS_DIR}

Run the evaluator first, or pass --results-dir DIR.
EOF
  exit 1
fi

if [[ ${ALLOW_EMPTY} -ne 1 ]]; then
  if ! find "${RESULTS_DIR}" -type f -print -quit 2>/dev/null | grep -q .; then
    cat >&2 <<EOF
ERROR: results directory contains no files:
  ${RESULTS_DIR}

Refusing to sync an empty results directory. Use --allow-empty to override.
EOF
    exit 1
  fi
fi

CMD=(uvx hf buckets sync "${RESULTS_DIR}" "${DEST_URI}")
if [[ ${DELETE} -eq 1 ]]; then
  CMD+=(--delete)
fi
if [[ ${DRY_RUN} -eq 1 ]]; then
  CMD+=(--dry-run)
fi
CMD+=("${EXTRA_ARGS[@]}")

echo "==> Source results dir: ${RESULTS_DIR}"
echo "==> Destination: ${DEST_URI}"
if [[ ${DRY_RUN} -eq 1 ]]; then
  echo "==> Dry run: enabled"
fi
if [[ ${DELETE} -eq 1 ]]; then
  echo "==> Remote delete: enabled"
else
  echo "==> Remote delete: disabled"
fi
printf '==> Running:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
