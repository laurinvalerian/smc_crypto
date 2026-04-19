#!/usr/bin/env bash
# Fire a Telegram alert on bot failures. Env-gated — silent no-op if
# TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID are not set, so enabling alerts
# is a one-shot edit to .env with zero code changes.
#
# Usage:
#   scripts/telegram_alert.sh "subject" "body"
#   Or via systemd: OnFailure=telegram-alert.service → templated systemd unit.
#
# Setup (two env vars in /root/bot/.env):
#   TELEGRAM_BOT_TOKEN=123456:ABC-DEF...   # from @BotFather
#   TELEGRAM_CHAT_ID=987654321             # user or group chat ID
#
# Test:
#   scripts/telegram_alert.sh "test" "hello from bongus bot"

set -u
set -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
cd "${REPO_ROOT}"

: "${ALERT_LOG:=logs/telegram_alert.log}"
mkdir -p "$(dirname "${ALERT_LOG}")"
stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { printf '[%s] %s\n' "$(stamp)" "$*" >> "${ALERT_LOG}"; }

# Load .env (bash-safe parse — only KEY=VAL lines, no exec).
if [[ -f .env ]]; then
  while IFS='=' read -r key value; do
    # skip comments/blanks, strip quotes
    [[ -z "$key" || "$key" == \#* ]] && continue
    value="${value%\"}"; value="${value#\"}"
    value="${value%\'}"; value="${value#\'}"
    export "$key=$value"
  done < <(grep -E '^[A-Z_][A-Z0-9_]*=' .env)
fi

if [[ -z "${TELEGRAM_BOT_TOKEN:-}" || -z "${TELEGRAM_CHAT_ID:-}" ]]; then
  log "alert skipped: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set in .env"
  exit 0
fi

SUBJECT="${1:-Bongus alert}"
BODY="${2:-(no body)}"
HOST="$(hostname)"

# Telegram max is 4096 chars; trim defensively.
MESSAGE="$(printf '🚨 *%s*\n`%s`\n\n%s' "${SUBJECT}" "${HOST}" "${BODY}" | head -c 3900)"

RESPONSE="$(curl -fsS -m 10 -X POST \
  "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
  -d chat_id="${TELEGRAM_CHAT_ID}" \
  -d parse_mode="Markdown" \
  --data-urlencode text="${MESSAGE}" 2>&1)" || {
    log "alert failed: ${RESPONSE}"
    exit 1
  }

log "alert sent: ${SUBJECT}"
exit 0
