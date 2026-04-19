#!/usr/bin/env bash
# Nightly WAL-safe backup of the trade journal SQLite DB.
#
# Uses sqlite3's `.backup` command which is safe to run while another process
# holds the DB open (coordinates with WAL), unlike plain `cp` which can produce
# a torn copy. See https://www.sqlite.org/backup.html.
#
# Output:   archive/journal_backups/journal_YYYYMMDD_HHMM.db.gz
# Retention: 30 days (older backups auto-pruned at the end).
# Log:      logs/backup_journal.log
#
# Scheduled by scripts/journal-backup.timer.template (daily 04:00 UTC).

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
cd "${REPO_ROOT}"

: "${JOURNAL_DB:=trade_journal/journal.db}"
: "${BACKUP_DIR:=archive/journal_backups}"
: "${BACKUP_LOG:=logs/backup_journal.log}"
: "${RETENTION_DAYS:=30}"

mkdir -p "$(dirname "${BACKUP_LOG}")" "${BACKUP_DIR}"

stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { printf '[%s] %s\n' "$(stamp)" "$*" | tee -a "${BACKUP_LOG}"; }

log "backup_journal: starting (db=${JOURNAL_DB}, dst=${BACKUP_DIR})"

if [[ ! -f "${JOURNAL_DB}" ]]; then
  log "backup_journal: journal DB not found at ${JOURNAL_DB} — nothing to back up."
  exit 0
fi

TS="$(date -u +%Y%m%d_%H%M%S)"
OUT_DB="${BACKUP_DIR}/journal_${TS}.db"
OUT_GZ="${OUT_DB}.gz"

# Use sqlite3's online backup so concurrent WAL writes stay safe.
sqlite3 "${JOURNAL_DB}" ".backup '${OUT_DB}'"
BYTES_RAW="$(stat -c %s "${OUT_DB}")"
# -f: allow overwrite if a stale .gz from a crashed run is present.
gzip -9 -f "${OUT_DB}"
BYTES_GZ="$(stat -c %s "${OUT_GZ}")"

log "backup_journal: wrote ${OUT_GZ} (raw=${BYTES_RAW}B, gz=${BYTES_GZ}B)"

# Retention: delete gz backups older than RETENTION_DAYS.
PRUNED="$(find "${BACKUP_DIR}" -maxdepth 1 -name 'journal_*.db.gz' -mtime "+${RETENTION_DAYS}" -print -delete 2>/dev/null | wc -l)"
if [[ "${PRUNED}" -gt 0 ]]; then
  log "backup_journal: pruned ${PRUNED} file(s) older than ${RETENTION_DAYS} days"
fi

# Sanity: file count + total size
COUNT="$(ls -1 "${BACKUP_DIR}"/journal_*.db.gz 2>/dev/null | wc -l)"
TOTAL="$(du -sh "${BACKUP_DIR}" 2>/dev/null | awk '{print $1}')"
log "backup_journal: done. Archive holds ${COUNT} file(s), ${TOTAL} total."
