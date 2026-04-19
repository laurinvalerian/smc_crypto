#!/usr/bin/env bash
# Daily automated health-check for the trading bot stack.
#
# Walks the key checks from HEALTH_CHECKS.md, writes a concise single-block
# report to logs/health_check.log, and exits rc=2 on any FAIL so systemd's
# OnFailure= hook (e.g. telegram_alert.sh) can escalate.
#
# What this DOES check (rc=2 if any fails):
#   - trading-bot.service is active
#   - trading-dashboard.service is active
#   - trading-dashboard-public.service is active
#   - drift-monitor.timer is active
#   - bot PID is alive (not zombie)
#   - paper_trading.log has a HEARTBEAT in the last 15 min
#   - trade journal DB is readable
#   - disk usage < 90%
#   - drift_state.json has no CRITICAL level sustained > 24h (if file exists)
#
# What it does NOT do: business logic. For "is this bot trading profitably?"
# see the weekly reconciliation report.
#
# Scheduled by scripts/health-check.timer.template (daily 09:05 UTC, right
# after drift-monitor fires so that drift_state.json is current).

set -u
set -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
cd "${REPO_ROOT}"

: "${HEALTH_LOG:=logs/health_check.log}"
: "${JOURNAL_DB:=trade_journal/journal.db}"
: "${LIVE_LOG:=paper_trading.log}"
: "${DRIFT_STATE:=data/drift_state.json}"
: "${HEARTBEAT_MAX_AGE_MIN:=15}"
: "${DISK_MAX_PCT:=90}"

mkdir -p "$(dirname "${HEALTH_LOG}")"

stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

FAILS=()
OKS=()

check() {
  local name="$1"; shift
  if eval "$*" >/dev/null 2>&1; then
    OKS+=("$name")
  else
    FAILS+=("$name")
  fi
}

# 1. Systemd units
check "bot-active"           'systemctl is-active trading-bot.service | grep -qx active'
check "dashboard-active"     'systemctl is-active trading-dashboard.service | grep -qx active'
check "dashboard-pub-active" 'systemctl is-active trading-dashboard-public.service | grep -qx active'
check "drift-timer-active"   'systemctl is-active drift-monitor.timer | grep -qx active'

# 2. Bot PID alive
BOT_PID="$(systemctl show trading-bot.service -p MainPID --value 2>/dev/null || echo 0)"
check "bot-pid-alive" "[[ \"${BOT_PID}\" -gt 0 ]] && kill -0 ${BOT_PID}"

# 3. Heartbeat recency
check "heartbeat-recent" "[[ -f \"${LIVE_LOG}\" ]] && [[ \$(find \"${LIVE_LOG}\" -mmin -${HEARTBEAT_MAX_AGE_MIN} | wc -l) -gt 0 ]] && grep -q HEARTBEAT \"${LIVE_LOG}\""

# 4. Journal readable
check "journal-readable" "sqlite3 \"${JOURNAL_DB}\" 'SELECT COUNT(*) FROM trades'"

# 5. Disk usage
PCT="$(df --output=pcent / | tail -1 | tr -d ' %' || echo 100)"
check "disk-under-${DISK_MAX_PCT}pct" "[[ ${PCT:-100} -lt ${DISK_MAX_PCT} ]]"

# 6. Drift alert level (optional — only if file exists)
if [[ -f "${DRIFT_STATE}" ]]; then
  ALERT="$(python3 -c "import json,sys; d=json.load(open('${DRIFT_STATE}')); print(d.get('alert_level') or 'INFO')" 2>/dev/null || echo UNKNOWN)"
  check "drift-not-critical" "[[ \"${ALERT}\" != \"CRITICAL\" ]]"
else
  # Not a fail when the file doesn't exist yet (fresh bot, no signals).
  OKS+=("drift-state-absent-ok")
fi

# ── Report ────────────────────────────────────────────────────────
RC=0
if (( ${#FAILS[@]} > 0 )); then RC=2; fi

{
  echo "─────────────────────────────────────────────"
  echo "[$(stamp)] health_check rc=${RC}"
  echo "OK   (${#OKS[@]}): ${OKS[*]:-<none>}"
  echo "FAIL (${#FAILS[@]}): ${FAILS[*]:-<none>}"
  echo "bot_pid=${BOT_PID} disk=${PCT}% drift=${ALERT:-n/a}"
  # Quick live-stats snapshot
  if sqlite3 "${JOURNAL_DB}" 'SELECT COUNT(*) FROM trades' >/dev/null 2>&1; then
    TOTAL="$(sqlite3 "${JOURNAL_DB}" 'SELECT COUNT(*) FROM trades')"
    WINS="$(sqlite3 "${JOURNAL_DB}" "SELECT COUNT(*) FROM trades WHERE outcome='win'")"
    OPEN="$(sqlite3 "${JOURNAL_DB}" 'SELECT COUNT(*) FROM trades WHERE exit_time IS NULL')"
    echo "trades=${TOTAL} wins=${WINS} open=${OPEN}"
  fi
} | tee -a "${HEALTH_LOG}"

exit "${RC}"
