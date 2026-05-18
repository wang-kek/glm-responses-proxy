#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SOURCE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
DEFAULT_TARGET_DIR="$(cd "$SOURCE_DIR/.." && pwd)/glm-responses-proxy-test-run"

TARGET_DIR="$DEFAULT_TARGET_DIR"
CLEAN_LOGS=1

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [--keep-logs] [--target DIR]

Options:
  --keep-logs    Preserve test-run logs, request capture log, and pid files after sync.
  --target DIR   Override the default test-run directory.
                 Default: $DEFAULT_TARGET_DIR
  -h, --help     Show this help message.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --keep-logs)
      CLEAN_LOGS=0
      shift
      ;;
    --target)
      [ $# -ge 2 ] || { echo "missing value for --target" >&2; exit 1; }
      TARGET_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

mkdir -p "$TARGET_DIR"

rsync -a \
  --delete \
  --exclude '.run/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  --exclude 'testhi.log' \
  "$SOURCE_DIR/" "$TARGET_DIR/"

mkdir -p "$TARGET_DIR/.run/logs" "$TARGET_DIR/.run/pids"
touch "$TARGET_DIR/testhi.log"

if [ "$CLEAN_LOGS" = "1" ]; then
  find "$TARGET_DIR/.run/logs" -type f -delete
  find "$TARGET_DIR/.run/pids" -type f -delete
  : > "$TARGET_DIR/testhi.log"
fi

echo "synced source to test run directory"
echo "source: $SOURCE_DIR"
echo "target: $TARGET_DIR"
if [ "$CLEAN_LOGS" = "1" ]; then
  echo "logs: cleaned"
else
  echo "logs: preserved"
fi
