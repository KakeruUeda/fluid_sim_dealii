#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./remote.conf

ssh "$HOST" "mkdir -p '$REMOTE_DIR'"

RSYNC_EXCLUDES=(
  --exclude ".git/"
  --exclude "__pycache__/"
  --exclude ".venv/"
  --exclude "outputs/"
  --exclude "build/"
)

rsync -avz --delete "${RSYNC_EXCLUDES[@]}" \
  "$LOCAL_ROOT"/ "$HOST":"$REMOTE_DIR"/

echo "[push] Done -> $HOST:$REMOTE_DIR"
