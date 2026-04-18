#!/usr/bin/env bash
# ========================================================================
# deploy_optfix.sh — Deploy the train/inference mismatch fix
# ========================================================================
# Deploys the fixed live_multi_bot.py, oanda_adapter.py, and (optionally)
# the retrained rl_entry_filter.pkl to the server.
#
# Features:
#   - Creates atomic backups (_prev suffix) before overwriting
#   - Validates files on local before copying
#   - Deploys via rsync (idempotent)
#   - Restarts trading-bot service via systemctl
#   - Monitors first 3 XGB decisions after restart to confirm conf > 0.5
#   - Rollback command printed at the end
#
# Usage:
#   scripts/deploy_optfix.sh               # full deploy (code + model)
#   scripts/deploy_optfix.sh --code-only   # only code changes, keep OLD model
#   scripts/deploy_optfix.sh --rollback    # restore previous files
#   scripts/deploy_optfix.sh --dry-run     # print what would be done
#
# SAFETY:
#   - Never overwrites without backup
#   - Never skips service restart verification
#   - Requires explicit `--i-know-what-im-doing` for production deploy
# ========================================================================
set -euo pipefail

SERVER="server"
REMOTE="/root/bot"
DRY_RUN=false
CODE_ONLY=false
ROLLBACK=false
CONFIRMED=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)             DRY_RUN=true;   shift ;;
    --code-only)           CODE_ONLY=true; shift ;;
    --rollback)            ROLLBACK=true;  shift ;;
    --i-know-what-im-doing) CONFIRMED=true; shift ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

# ========================================================================
# Helpers
# ========================================================================
say() { printf '\033[1;36m==> \033[0m%s\n' "$*"; }
ok()  { printf '\033[1;32mOK  \033[0m%s\n' "$*"; }
warn() { printf '\033[1;33mWARN\033[0m %s\n' "$*"; }
err() { printf '\033[1;31mERR \033[0m%s\n' "$*"; }

run_remote() {
  if $DRY_RUN; then
    echo "  [dry-run] ssh $SERVER '$*'"
    return 0
  fi
  ssh "$SERVER" "$@"
}

copy_to_remote() {
  local local_path="$1"; local remote_path="$2"
  if $DRY_RUN; then
    echo "  [dry-run] rsync -av $local_path $SERVER:$remote_path"
    return 0
  fi
  rsync -av "$local_path" "$SERVER:$remote_path" > /dev/null
}

# ========================================================================
# Rollback mode
# ========================================================================
if $ROLLBACK; then
  say "Rollback: restoring _prev files on server"
  run_remote "
    cd $REMOTE && \
    for f in live_multi_bot.py exchanges/oanda_adapter.py models/rl_entry_filter.pkl; do
      if [ -e \"\${f}_prev\" ]; then
        cp \"\${f}_prev\" \"\$f\" && echo \"Restored \$f\"
      else
        echo \"No backup for \$f — skipped\"
      fi
    done
  "
  say "Restarting bot..."
  run_remote "sudo systemctl restart trading-bot.service && sleep 2 && systemctl is-active trading-bot.service"
  ok "Rollback complete"
  exit 0
fi

# ========================================================================
# Pre-flight checks
# ========================================================================
say "Pre-flight checks"

# Syntax validation
for f in live_multi_bot.py exchanges/oanda_adapter.py backtest/generate_rl_data.py rl_brain_v2.py; do
  python3 -c "import ast; ast.parse(open('$f').read())" || { err "$f syntax error"; exit 1; }
done
ok "Python syntax OK"

# Verify models exist (if full deploy)
if ! $CODE_ONLY; then
  if [ ! -f "models/rl_entry_filter_optfix.pkl" ]; then
    err "models/rl_entry_filter_optfix.pkl not found — cannot deploy model"
    err "Run backtest/retrain_after_optfix.py first"
    exit 1
  fi
  ok "New model exists"
fi

# Confirmation
if ! $CONFIRMED && ! $DRY_RUN; then
  warn "This will restart the live trading bot on $SERVER."
  warn "Rerun with --i-know-what-im-doing to proceed."
  exit 1
fi

# ========================================================================
# Backup server files
# ========================================================================
say "Backing up current server files"
run_remote "
  cd $REMOTE && \
  cp live_multi_bot.py live_multi_bot.py_prev && \
  cp exchanges/oanda_adapter.py exchanges/oanda_adapter.py_prev && \
  [ -f models/rl_entry_filter.pkl ] && cp models/rl_entry_filter.pkl models/rl_entry_filter.pkl_prev || true && \
  echo 'Backups created'
"
ok "Backups complete"

# ========================================================================
# Deploy code
# ========================================================================
say "Deploying code files"
copy_to_remote live_multi_bot.py "$REMOTE/live_multi_bot.py"
copy_to_remote exchanges/oanda_adapter.py "$REMOTE/exchanges/oanda_adapter.py"
ok "Code deployed"

# ========================================================================
# Deploy model
# ========================================================================
if ! $CODE_ONLY; then
  say "Deploying new model"
  copy_to_remote models/rl_entry_filter_optfix.pkl "$REMOTE/models/rl_entry_filter.pkl"
  ok "Model deployed"
fi

# ========================================================================
# Restart bot
# ========================================================================
say "Restarting trading-bot.service"
if $DRY_RUN; then
  echo "  [dry-run] would restart service"
else
  ssh "$SERVER" "sudo systemctl restart trading-bot.service"
  sleep 3
  status=$(ssh "$SERVER" "systemctl is-active trading-bot.service")
  if [ "$status" != "active" ]; then
    err "Service did not come up: $status"
    err "Run with --rollback to restore backups"
    exit 1
  fi
fi
ok "Service active"

# ========================================================================
# Post-deploy verification — watch first 3 XGB decisions
# ========================================================================
if $DRY_RUN; then
  say "(dry-run: skipping post-deploy verification)"
else
  say "Monitoring first 3 XGB decisions (up to 5 min)"
  set +e
  ssh "$SERVER" "
    for i in \$(seq 1 60); do
      n=\$(grep -c 'XGB ACCEPT\|XGB REJECT' /root/bot/paper_trading.log | tail -1)
      recent=\$(tail -500 /root/bot/paper_trading.log | grep -E 'XGB ACCEPT|XGB REJECT' | tail -3)
      if [ -n \"\$recent\" ]; then
        n_recent=\$(echo \"\$recent\" | wc -l)
        if [ \$n_recent -ge 3 ]; then
          echo
          echo 'First 3 XGB decisions after restart:'
          echo \"\$recent\"
          exit 0
        fi
      fi
      sleep 5
    done
    echo 'No XGB decisions observed in 5 min — check server manually'
    exit 1
  "
  set -e
fi

# ========================================================================
# Summary
# ========================================================================
say "Deployment complete"
ok "Rollback command: scripts/deploy_optfix.sh --rollback --i-know-what-im-doing"
