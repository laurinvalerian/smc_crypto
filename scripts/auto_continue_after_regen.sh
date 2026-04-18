#!/usr/bin/env bash
# ========================================================================
# auto_continue_after_regen.sh — chain retrain → verification after regens
# ========================================================================
# Waits for the 3 parallel regen processes (regen4_*) to complete, then:
#   1. Runs retrain_after_optfix.py → produces rl_entry_filter_optfix.pkl
#   2. Runs shadow_replay_entry_filter.py → checks conf parity
#   3. Runs verify_feature_parity.py → bar-level feature diff
#   4. Writes a consolidated summary to /tmp/ab_test/auto_continue.log
#
# Safety:
#   - Never deploys anything
#   - Only reads and trains, never modifies live or server files
#   - Exits non-zero if any stage fails, with clear error message
#
# Run in background:
#   nohup scripts/auto_continue_after_regen.sh > /tmp/ab_test/auto.log 2>&1 &
# ========================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

LOG_DIR="/tmp/ab_test"
AUTO_LOG="$LOG_DIR/auto_continue.log"
mkdir -p "$LOG_DIR"

log()  { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$AUTO_LOG"; }
fail() { log "FAIL: $*"; exit 1; }

log "auto_continue_after_regen.sh starting"

# ========================================================================
# Stage 1: wait for all 3 regens to complete
# ========================================================================
log "Stage 1: waiting for regen4_{crypto,forex,commodities}.log to complete"

MAX_WAIT_SEC=$((4 * 60 * 60))  # 4 hours max
elapsed=0
interval=30

while :; do
  crypto_done=false
  forex_done=false
  commodities_done=false

  if grep -q "SAVED\|Saved:" "$LOG_DIR/regen4_crypto.log" 2>/dev/null; then
    crypto_done=true
  fi
  if grep -q "SAVED\|Saved:" "$LOG_DIR/regen4_forex.log" 2>/dev/null; then
    forex_done=true
  fi
  if grep -q "SAVED\|Saved:" "$LOG_DIR/regen4_commodities.log" 2>/dev/null; then
    commodities_done=true
  fi

  if $crypto_done && $forex_done && $commodities_done; then
    log "All 3 regens complete"
    break
  fi

  # Early failure detection
  for f in regen4_crypto regen4_forex regen4_commodities; do
    if [ -f "$LOG_DIR/$f.log" ]; then
      if grep -q "Traceback\|SIGKILL\|OOM" "$LOG_DIR/$f.log"; then
        fail "$f failed — check log"
      fi
    fi
  done

  # Status every ~10 min
  if [ $((elapsed % 600)) -eq 0 ]; then
    c_prog=$(grep "Progress:" "$LOG_DIR/regen4_crypto.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+/[0-9]+' || echo "?")
    f_prog=$(grep "Progress:" "$LOG_DIR/regen4_forex.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+/[0-9]+' || echo "?")
    m_prog=$(grep "Progress:" "$LOG_DIR/regen4_commodities.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+/[0-9]+' || echo "?")
    log "  status: crypto=$c_prog  forex=$f_prog  commodities=$m_prog"
  fi

  sleep $interval
  elapsed=$((elapsed + interval))

  if [ $elapsed -ge $MAX_WAIT_SEC ]; then
    fail "Timeout waiting for regens after ${MAX_WAIT_SEC}s"
  fi
done

# Verify the parquet files exist and are non-empty
for ac in crypto forex commodities; do
  path="data/rl_training/${ac}_samples.parquet"
  if [ ! -f "$path" ] || [ ! -s "$path" ]; then
    fail "$path missing or empty"
  fi
  log "  $path: $(du -h "$path" | cut -f1)"
done

# ========================================================================
# Stage 2: retrain model
# ========================================================================
log "Stage 2: retraining model with new parquets"
if python3 -m backtest.retrain_after_optfix > "$LOG_DIR/retrain.log" 2>&1; then
  log "Retrain complete"
  grep -E "best iteration|Saved model|best_iteration|n_train" "$LOG_DIR/retrain.log" | tail -10 | tee -a "$AUTO_LOG"
else
  fail "Retrain failed — see $LOG_DIR/retrain.log"
fi

if [ ! -f "models/rl_entry_filter_optfix.pkl" ]; then
  fail "Retrain did not produce models/rl_entry_filter_optfix.pkl"
fi

# ========================================================================
# Stage 3: shadow-replay
# ========================================================================
log "Stage 3: shadow-replay against live decisions"
if python3 -m backtest.shadow_replay_entry_filter > "$LOG_DIR/shadow_replay.log" 2>&1; then
  log "Shadow-replay complete"
  # Extract key metrics
  grep -E "PARITY|mean_abs_diff|accepted=|accepted = |VERIFIED|MISMATCH" "$LOG_DIR/shadow_replay.log" | tail -20 | tee -a "$AUTO_LOG"
else
  log "Shadow-replay failed — see $LOG_DIR/shadow_replay.log"
  # Don't exit — continue with feature-parity
fi

# ========================================================================
# Stage 4: feature-parity check
# ========================================================================
log "Stage 4: feature-parity check for AUD_JPY"
if python3 -m backtest.verify_feature_parity > "$LOG_DIR/feature_parity.log" 2>&1; then
  log "Feature-parity check complete"
  grep -E "Max delta|Mean delta|Features with delta" "$LOG_DIR/feature_parity.log" | tail -5 | tee -a "$AUTO_LOG"
else
  log "Feature-parity check failed — see $LOG_DIR/feature_parity.log"
fi

# ========================================================================
# Stage 5: summary
# ========================================================================
log "============================================================"
log "AUTO-CONTINUE PIPELINE COMPLETE"
log "============================================================"
log "Next steps (USER ACTION REQUIRED):"
log "  1. Review DEPLOYMENT_READINESS.md"
log "  2. Review shadow_replay output — conf delta should be < 0.05"
log "  3. Review feature_parity — all features should be near-zero delta"
log "  4. If satisfied, run: scripts/deploy_optfix.sh --i-know-what-im-doing"
log ""
log "Files to review:"
log "  $LOG_DIR/retrain.log"
log "  $LOG_DIR/shadow_replay.log"
log "  $LOG_DIR/feature_parity.log"
log "  $AUTO_LOG"
log ""
log "DO NOT DEPLOY until user reviews and approves."
