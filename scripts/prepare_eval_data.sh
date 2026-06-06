#!/usr/bin/env bash
set -euo pipefail

# Prepare the sampled USTBench QA evaluation data used by this fork.
#
# Default behavior:
#   1. Download zhangdw/USTBench-ST-Planning-10pct from Hugging Face via `uv run hf` into a stable temporary cache.
#   2. Copy its USTBench-compatible question_answering/Data tree into
#      data/, which is the default path read by the lightweight QA evaluator.
#
# The target directory is protected by default. Use --force to replace an
# existing data/ directory.

DATASET_REPO="zhangdw/USTBench-ST-Planning-10pct"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOWNLOAD_DIR=""
TARGET_DIR="${REPO_ROOT}/data"
FORCE=0
DEFAULT_DOWNLOAD_DIR=1

abs_path() {
  python3 - "$1" <<'PYABS'
import os
import sys
print(os.path.abspath(sys.argv[1]))
PYABS
}

usage() {
  cat <<EOF
Usage: $0 [options]

Download the sampled USTBench evaluation dataset from Hugging Face and install
it into the USTBench question-answering data path.

Options:
  --repo REPO_ID          Hugging Face dataset repo.
                          Default: ${DATASET_REPO}
  --download-dir DIR      Local dataset download/cache directory.
                          Default: a stable directory under \${TMPDIR:-/tmp}
                          so interrupted downloads can be resumed by re-running.
  --target-dir DIR        Destination data directory used by USTBench.
                          Default: ${TARGET_DIR}
  --force                 Replace an existing non-empty target directory.
  -h, --help              Show this help.

Examples:
  bash scripts/prepare_eval_data.sh
  bash scripts/prepare_eval_data.sh --force
  bash scripts/prepare_eval_data.sh --repo zhangdw/USTBench-ST-Planning-10pct

After running, QA evaluation can read files like:
  data/urban_planning/planning_QA.json
  data/congestion_prediction/st_understanding_QA.json
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      DATASET_REPO="${2:?Missing value for --repo}"
      shift 2
      ;;
    --download-dir)
      DOWNLOAD_DIR="$(abs_path "${2:?Missing value for --download-dir}")"
      DEFAULT_DOWNLOAD_DIR=0
      shift 2
      ;;
    --target-dir)
      TARGET_DIR="$(abs_path "${2:?Missing value for --target-dir}")"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if command -v uv >/dev/null 2>&1; then
  HF_CMD=(uv run hf)
elif command -v hf >/dev/null 2>&1; then
  HF_CMD=(hf)
else
  echo "ERROR: uv is required for the default workflow, or install the hf CLI." >&2
  exit 1
fi

if [[ -z "${DOWNLOAD_DIR}" ]]; then
  TMP_DOWNLOAD_PARENT="${TMPDIR:-/tmp}"
  TMP_DOWNLOAD_PARENT="${TMP_DOWNLOAD_PARENT%/}"
  if [[ -z "${TMP_DOWNLOAD_PARENT}" ]]; then
    TMP_DOWNLOAD_PARENT="/tmp"
  fi
  REPO_CACHE_NAME="${DATASET_REPO//\//__}"
  DOWNLOAD_DIR="${TMP_DOWNLOAD_PARENT}/ustbench-data/${REPO_CACHE_NAME}"
fi
mkdir -p "${DOWNLOAD_DIR}"

echo "==> Repository root: ${REPO_ROOT}"
echo "==> HF dataset repo: ${DATASET_REPO}"
echo "==> Download/cache dir: ${DOWNLOAD_DIR}"
if [[ ${DEFAULT_DOWNLOAD_DIR} -eq 1 ]]; then
  echo "==> Download/cache dir is outside the repository and is kept for resumable re-runs."
fi
echo "==> Target data dir: ${TARGET_DIR}"

echo "==> Downloading dataset with: ${HF_CMD[*]} download ${DATASET_REPO} --repo-type dataset"
"${HF_CMD[@]}" download "${DATASET_REPO}" \
  --repo-type dataset \
  --local-dir "${DOWNLOAD_DIR}"

SOURCE_DIR="${DOWNLOAD_DIR}/question_answering/Data"
if [[ ! -d "${SOURCE_DIR}" ]]; then
  echo "ERROR: downloaded dataset does not contain expected directory: ${SOURCE_DIR}" >&2
  exit 1
fi

if [[ -e "${TARGET_DIR}" ]]; then
  if [[ ${FORCE} -ne 1 ]]; then
    if find "${TARGET_DIR}" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
      cat >&2 <<EOF
ERROR: target directory already exists and is non-empty:
  ${TARGET_DIR}

Re-run with --force to replace it, or choose a different --target-dir.
EOF
      exit 1
    fi
  else
    echo "==> Removing existing target directory because --force was provided."
    rm -rf "${TARGET_DIR}"
  fi
fi

mkdir -p "${TARGET_DIR}"
echo "==> Installing question_answering/Data into target path."
cp -a "${SOURCE_DIR}/." "${TARGET_DIR}/"

COUNTS_FILE="${DOWNLOAD_DIR}/metadata/counts.json"
if [[ -f "${COUNTS_FILE}" ]]; then
  echo "==> Dataset counts:"
  python3 - "${COUNTS_FILE}" <<'PY'
import json
import sys
from pathlib import Path
counts = json.loads(Path(sys.argv[1]).read_text())
print(f"total_sampled_cases: {counts.get('total_sampled_cases')}")
for subset, n in sorted(counts.get('sample_counts_by_subset', {}).items()):
    print(f"{subset}: {n}")
PY
fi

echo "==> Done. USTBench QA data is ready at: ${TARGET_DIR}"
cat <<EOF

Suggested evaluation subsets:
  st_understanding
  planning

Example:
  uv run python scripts/evaluate_qa_vllm.py \
    --model <served-model-name> \
    --base-url http://127.0.0.1:8000/v1 \
    --api-key EMPTY \
    --datasets "st_understanding,planning" \
    --batch-size 32
EOF
