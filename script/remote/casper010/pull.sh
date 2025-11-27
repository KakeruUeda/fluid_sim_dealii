#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./remote.conf

rsync -avz --info=progress2 --partial \
  casper010:"$REMOTE_OUTPUT_DIR"/ \
  "$LOCAL_OUTPUT_DIR"/