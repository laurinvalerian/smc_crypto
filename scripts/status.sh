#!/usr/bin/env bash
# ========================================================================
# status.sh — one-command status overview for the ongoing fix work
# ========================================================================
# Shows at a glance:
#   - Regen progress (local)
#   - Auto-continue daemon status
#   - Retrain/shadow-replay results (if complete)
#   - Live bot health (via SSH)
#   - Current AUD_USD trade state
#   - Recent XGB decisions
# ========================================================================
cd "$(dirname "$0")/.."

bold() { printf '\033[1m%s\033[0m' "$*"; }
c_ok() { printf '\033[32m%s\033[0m' "$*"; }
c_warn() { printf '\033[33m%s\033[0m' "$*"; }
c_err() { printf '\033[31m%s\033[0m' "$*"; }
c_info() { printf '\033[36m%s\033[0m' "$*"; }

hr() { printf '\033[2m%.s─\033[0m' {1..72}; printf '\n'; }

echo
echo "$(bold '╔══════════════════════════════════════════════════════════════════╗')"
echo "$(bold '║  Train/Inference Mismatch Fix — Status Dashboard                 ║')"
echo "$(bold "║  $(date)                                  ║")"
echo "$(bold '╚══════════════════════════════════════════════════════════════════╝')"
echo

# ── Regen progress ────────────────────────────────────────────────────
echo "$(c_info '▶ Regen progress (/tmp/ab_test/regen4_*.log)')"
hr
for ac in crypto forex commodities; do
  log="/tmp/ab_test/regen4_${ac}.log"
  if [ ! -f "$log" ]; then
    printf "  %-15s %s\n" "$ac" "$(c_err 'NOT STARTED')"
    continue
  fi
  prog=$(grep "Progress:" "$log" 2>/dev/null | tail -1 | grep -oE '[0-9]+/[0-9]+ \([0-9]+%\)' || echo "")
  saved=$(grep "SUMMARY\|Saved:" "$log" 2>/dev/null | tail -1)
  if [ -n "$saved" ]; then
    printf "  %-15s %s\n" "$ac" "$(c_ok '✓ DONE')"
  elif [ -n "$prog" ]; then
    printf "  %-15s %s\n" "$ac" "$(c_info "$prog")"
  else
    printf "  %-15s %s\n" "$ac" "$(c_warn '(no progress yet)')"
  fi
done
echo

# ── Auto-continue daemon ──────────────────────────────────────────────
echo "$(c_info '▶ Auto-continue daemon')"
hr
daemon_pid=$(pgrep -f "auto_continue_after_regen" | head -1)
if [ -n "$daemon_pid" ]; then
  printf "  pid=%s %s\n" "$daemon_pid" "$(c_ok '(running)')"
  echo "  Recent log:"
  tail -5 /tmp/ab_test/auto_continue.log 2>/dev/null | sed 's/^/    /'
else
  printf "  %s\n" "$(c_warn 'not running')"
fi
echo

# ── Retrain / shadow-replay ───────────────────────────────────────────
echo "$(c_info '▶ Pipeline outputs')"
hr
for f in "models/rl_entry_filter_optfix.pkl" \
         "backtest/results/shadow_replay_results.json" \
         "backtest/results/ab_test_results.json"; do
  if [ -f "$f" ]; then
    age=$(stat -f "%m" "$f" 2>/dev/null || stat -c "%Y" "$f" 2>/dev/null)
    now=$(date +%s)
    mins=$(( (now - age) / 60 ))
    printf "  %s %-45s %s\n" "$(c_ok '✓')" "$f" "(${mins}m old)"
  else
    printf "  %s %s\n" "$(c_warn '○')" "$f"
  fi
done
echo

# ── Live bot health ──────────────────────────────────────────────────
echo "$(c_info '▶ Live bot (server)')"
hr
ssh server 'systemctl is-active trading-bot.service' 2>/dev/null | sed 's/^/  service: /'
ssh server 'grep HEARTBEAT /root/bot/paper_trading.log | tail -1' 2>/dev/null | sed 's/^/  /'
echo

# ── AUD_USD trade ────────────────────────────────────────────────────
echo "$(c_info '▶ Open positions')"
hr
ssh server 'cd /root/bot && sqlite3 trade_journal/journal.db "SELECT trade_id, symbol, direction, entry_time, ROUND(entry_price,5), ROUND(sl_original,5), ROUND(tp,5), COALESCE(outcome,\"OPEN\") FROM trades WHERE exit_time IS NULL ORDER BY entry_time DESC;"' 2>/dev/null | while IFS='|' read -r tid sym dir time entry sl tp outcome; do
  if [ -n "$tid" ]; then
    printf "  #%s %s %s entry=%s SL=%s TP=%s [%s]\n" "$tid" "$sym" "$dir" "$entry" "$sl" "$tp" "$outcome"
  fi
done
echo

# ── Recent XGB decisions ─────────────────────────────────────────────
echo "$(c_info '▶ Last 5 XGB decisions (server)')"
hr
ssh server 'grep -E "XGB (ACCEPT|REJECT)" /root/bot/paper_trading.log | tail -5' 2>/dev/null | sed 's/^/  /'
echo

# ── Task list ────────────────────────────────────────────────────────
echo "$(c_info '▶ Task list')"
hr
echo "  (run from Claude Code session to see live task states)"
echo

# ── Quick actions ─────────────────────────────────────────────────────
echo "$(c_info '▶ Quick actions')"
hr
echo "  View regen logs:       tail -f /tmp/ab_test/regen4_{crypto,forex,commodities}.log"
echo "  View auto-continue:    tail -f /tmp/ab_test/auto_continue.log"
echo "  Deploy (dry-run):      scripts/deploy_optfix.sh --dry-run"
echo "  Deploy (real):         scripts/deploy_optfix.sh --i-know-what-im-doing"
echo "  Rollback:              scripts/deploy_optfix.sh --rollback --i-know-what-im-doing"
echo "  Full report:           less DEPLOYMENT_READINESS.md"
echo
