#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./remote.conf

SESSION="fluid_sim"

# --- without tmux ---
# ssh "$HOST" /bin/zsh -l -c "
#   $RUN_CMD
# "

# --- run using tmux ---
ssh "$HOST" /bin/zsh -l -c "
  /opt/homebrew/bin/tmux has-session -t \"$SESSION\" 2>/dev/null || /opt/homebrew/bin/tmux new -d -s \"$SESSION\"
  /opt/homebrew/bin/tmux send-keys -t \"$SESSION\" $(printf '%q' "$RUN_CMD") C-m
"
echo "Remote job started in session: $SESSION"

# --- COMMAND IN TMUX -------------------------------------
# reattach to the session
# -> ssh -t macpro /bin/zsh -l -c 'export ITERM_SHELL_INTEGRATION_INSTALLED=; export ITERM_SESSION_ID=; /opt/homebrew/bin/tmux attach -t "fluid_sim"'
# exit from the session
# -> ctr-b + d
# kill the session
# -> ctr-c
# ----------------------------------------------------------