#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./remote.conf
SESSION="fluid_sim_sampleX_bc_from_sampleC"

# --- run using tmux ---
ssh "$HOST" "
  tmux has-session -t \"$SESSION\" 2>/dev/null || tmux new -d -s \"$SESSION\"
  tmux send-keys -t \"$SESSION\" $(printf '%q' "$RUN_CMD") C-m
"
echo "Remote job started in session: $SESSION"

# --- COMMANDS FOR TMUX SESSION ---------------
# attach (check log):
#   ssh -t casper010 "tmux attach -t {name_of_job}"
#
# detach:
#   Ctrl-b + d 
#
# kill the session:
#   Ctrl-c
# ---------------------------------------------
