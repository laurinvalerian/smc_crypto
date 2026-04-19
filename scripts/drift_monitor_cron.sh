#!/usr/bin/env bash
# Drift-monitor standalone runner for cron / launchd / systemd.
#
# Wraps `python3 -m drift_monitor` as a single-shot check. Complements the
# in-bot async loop (drift_monitor.run_drift_monitor) — this external runner
# acts as a safety net in case the bot loop is paused, restarting, or the
# bot process is temporarily down. The external run is non-invasive: it
# opens the journal DB in read-only mode and writes only to the state file.
#
# Alerts:
#   - logs to $DRIFT_LOG (default: logs/drift_monitor_cron.log)
#   - writes machine-readable state to data/drift_state.json (same file
#     the bot loop writes; external run overwrites / updates it)
#   - exits with rc=2 when a CRITICAL/WARNING drift is detected so cron
#     wrappers can escalate (e.g. launchd's StandardErrorPath, systemd's
#     OnFailure=... hooks, or GitHub Actions assertions)
#
# Usage:
#   bash scripts/drift_monitor_cron.sh                    # run now
#   DRIFT_LOG=/tmp/drift.log bash scripts/drift_monitor_cron.sh
#
# Intended schedule: once per day during the 4-week pre-funded paper phase.
#   - macOS: scripts/drift_monitor.plist.template → ~/Library/LaunchAgents/
#   - server: scripts/drift-monitor.service.template +
#             scripts/drift-monitor.timer.template → /etc/systemd/system/

set -u
set -o pipefail

# ── Resolve repo root ─────────────────────────────────────────────
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
cd "${REPO_ROOT}"

# ── Inputs (env-overridable) ──────────────────────────────────────
: "${PYTHON_BIN:=python3}"
: "${DRIFT_DB:=trade_journal/journal.db}"
: "${DRIFT_STATE_FILE:=data/drift_state.json}"
: "${DRIFT_LOG:=logs/drift_monitor_cron.log}"

mkdir -p "$(dirname "${DRIFT_LOG}")"
mkdir -p "$(dirname "${DRIFT_STATE_FILE}")"

stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

log() {
  printf '[%s] %s\n' "$(stamp)" "$*" | tee -a "${DRIFT_LOG}"
}

log "drift_monitor_cron: starting (cwd=${REPO_ROOT}, db=${DRIFT_DB}, state=${DRIFT_STATE_FILE})"

if [[ ! -f "${DRIFT_DB}" ]]; then
  log "drift_monitor_cron: journal DB not found at ${DRIFT_DB} — nothing to check."
  exit 0
fi

# Run the standalone check. The module prints a JSON summary to stdout
# (via drift_monitor.main) that we capture in the log. It writes
# state_file with the full per-feature table.
set +e
OUTPUT="$("${PYTHON_BIN}" -m drift_monitor \
    --db "${DRIFT_DB}" \
    --state-file "${DRIFT_STATE_FILE}" 2>&1)"
RC=$?
set -e

printf '%s\n' "${OUTPUT}" | tee -a "${DRIFT_LOG}"

if [[ ${RC} -ne 0 ]]; then
  log "drift_monitor_cron: check exited rc=${RC} (non-alert failure, e.g. missing files)."
  exit "${RC}"
fi

# Parse alert_level out of the JSON summary without requiring jq.
# The summary line looks like: "alert_level": "CRITICAL",
ALERT_LEVEL="$(printf '%s\n' "${OUTPUT}" \
    | grep -o '"alert_level": "[A-Z]*"' \
    | head -1 \
    | sed 's/.*"\([A-Z]*\)".*/\1/' || true)"

case "${ALERT_LEVEL}" in
  CRITICAL|WARNING)
    log "drift_monitor_cron: ALERT — level=${ALERT_LEVEL}. See ${DRIFT_STATE_FILE} for per-feature table."
    exit 2
    ;;
  INFO|"")
    log "drift_monitor_cron: OK — no major drift."
    exit 0
    ;;
  *)
    log "drift_monitor_cron: unexpected alert_level='${ALERT_LEVEL}' — treating as INFO."
    exit 0
    ;;
esac
