#!/usr/bin/env bash
set -euo pipefail

# Direction: local USTBench artifacts -> Hugging Face bucket.
# Safe default: no remote deletion. Pass --delete only from an authoritative
# complete local copy; it deletes remote files absent locally for the selected
# artifact prefixes.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BUCKET_ID="${HF_BUCKET_ID:-zhangdw/leo-benchmark}"
REMOTE_PREFIX="${HF_USTBENCH_PREFIX:-USTBench}"
HF_CLI_STRING="${HF_CLI:-uvx hf}"
DRY_RUN=0
DELETE=0
IGNORE_EXISTING=0
EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Usage: hfsync/local_to_remote.sh [options] [-- extra hf sync args]

Direction:
  LOCAL USTBench artifacts -> HF bucket

Default sync pair:
  results/ -> hf://buckets/zhangdw/leo-benchmark/USTBench/results

Safe defaults:
  - Does not delete remote files absent locally.
  - May update same-path remote files if local files differ.
  - Use --ignore-existing / --new-only for strictly create-only behavior.

Options:
  --dry-run                 Print the sync plan without uploading.
  --delete                  Delete REMOTE files absent locally. Use carefully.
  --ignore-existing         Skip remote files that already exist; only upload new files.
  --new-only                Alias for --ignore-existing.
  --bucket BUCKET_ID        Bucket ID. Default: zhangdw/leo-benchmark.
  --prefix PREFIX           Remote prefix inside the bucket. Default: USTBench.
  -h, --help                Show this help.

Environment overrides:
  HF_BUCKET_ID              Same as --bucket.
  HF_USTBENCH_PREFIX        Same as --prefix.
  HF_CLI                    Command used to run hf. Default: "uvx hf".

Examples:
  hfsync/local_to_remote.sh --dry-run
  hfsync/local_to_remote.sh --new-only
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --delete)
      DELETE=1
      shift
      ;;
    --ignore-existing|--new-only)
      IGNORE_EXISTING=1
      shift
      ;;
    --bucket)
      [[ $# -ge 2 ]] || { echo "ERROR: --bucket requires a value" >&2; exit 2; }
      BUCKET_ID="$2"
      shift 2
      ;;
    --prefix)
      [[ $# -ge 2 ]] || { echo "ERROR: --prefix requires a value" >&2; exit 2; }
      REMOTE_PREFIX="$2"
      shift 2
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
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

REMOTE_PREFIX="${REMOTE_PREFIX#/}"
REMOTE_PREFIX="${REMOTE_PREFIX%/}"
read -r -a HF_CMD <<< "$HF_CLI_STRING"

LOCAL_DIR="${REPO_ROOT}/results"
if [[ -n "$REMOTE_PREFIX" ]]; then
  REMOTE_URI="hf://buckets/${BUCKET_ID}/${REMOTE_PREFIX}/results"
else
  REMOTE_URI="hf://buckets/${BUCKET_ID}/results"
fi

if [[ ! -d "$LOCAL_DIR" ]]; then
  echo "ERROR: local results directory does not exist: $LOCAL_DIR" >&2
  exit 1
fi

CMD=("${HF_CMD[@]}" buckets sync "$LOCAL_DIR" "$REMOTE_URI")
[[ "$DELETE" -eq 1 ]] && CMD+=(--delete)
[[ "$DRY_RUN" -eq 1 ]] && CMD+=(--dry-run)
[[ "$IGNORE_EXISTING" -eq 1 ]] && CMD+=(--ignore-existing)
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "==> Artifact: results"
echo "Direction: LOCAL -> REMOTE"
echo "Local:     $LOCAL_DIR"
echo "Remote:    $REMOTE_URI"
[[ "$DRY_RUN" -eq 1 ]] && echo "Mode:      dry-run" || echo "Mode:      apply"
[[ "$DELETE" -eq 1 ]] && echo "Delete:    enabled (REMOTE files absent locally may be deleted)" || echo "Delete:    disabled"
[[ "$IGNORE_EXISTING" -eq 1 ]] && echo "Existing:  skip remote-existing files" || echo "Existing:  update same-path remote files if changed"
printf 'Command:  '
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
