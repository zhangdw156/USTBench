#!/usr/bin/env bash
set -euo pipefail

# Direction: Hugging Face bucket -> local USTBench artifacts.
# Safe default: no local deletion. Pass --delete only when the remote prefix is
# the authoritative complete copy; it deletes local files absent remotely for
# the selected artifact prefixes.

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
Usage: hfsync/remote_to_local.sh [options] [-- extra hf sync args]

Direction:
  HF bucket -> LOCAL USTBench artifacts

Default sync pair:
  hf://buckets/zhangdw/leo-benchmark/USTBench/results -> results/

Safe defaults:
  - Does not delete local files absent remotely.
  - May update same-path local files if remote files differ.
  - Use --ignore-existing / --new-only for strictly create-only behavior.

Options:
  --dry-run                 Print the sync plan without downloading.
  --delete                  Delete LOCAL files absent remotely. Use carefully.
  --ignore-existing         Skip local files that already exist; only download new files.
  --new-only                Alias for --ignore-existing.
  --bucket BUCKET_ID        Bucket ID. Default: zhangdw/leo-benchmark.
  --prefix PREFIX           Remote prefix inside the bucket. Default: USTBench.
  -h, --help                Show this help.

Environment overrides:
  HF_BUCKET_ID              Same as --bucket.
  HF_USTBENCH_PREFIX        Same as --prefix.
  HF_CLI                    Command used to run hf. Default: "uvx hf".

Examples:
  hfsync/remote_to_local.sh --dry-run
  hfsync/remote_to_local.sh --new-only
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
mkdir -p "$LOCAL_DIR"

CMD=("${HF_CMD[@]}" buckets sync "$REMOTE_URI" "$LOCAL_DIR")
[[ "$DELETE" -eq 1 ]] && CMD+=(--delete)
[[ "$DRY_RUN" -eq 1 ]] && CMD+=(--dry-run)
[[ "$IGNORE_EXISTING" -eq 1 ]] && CMD+=(--ignore-existing)
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "==> Artifact: results"
echo "Direction: REMOTE -> LOCAL"
echo "Remote:    $REMOTE_URI"
echo "Local:     $LOCAL_DIR"
[[ "$DRY_RUN" -eq 1 ]] && echo "Mode:      dry-run" || echo "Mode:      apply"
[[ "$DELETE" -eq 1 ]] && echo "Delete:    enabled (LOCAL files absent remotely may be deleted)" || echo "Delete:    disabled"
[[ "$IGNORE_EXISTING" -eq 1 ]] && echo "Existing:  skip local-existing files" || echo "Existing:  update same-path local files if changed"
printf 'Command:  '
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
