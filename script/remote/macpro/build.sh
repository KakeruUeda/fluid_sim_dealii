#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./remote.conf

ssh "$HOST" /bin/zsh -l -c "
  $BUILD_CMD
"
