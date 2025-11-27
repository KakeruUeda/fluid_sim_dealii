#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./remote.conf

ssh "$HOST" "bash -l -c '$BUILD_CMD'"
