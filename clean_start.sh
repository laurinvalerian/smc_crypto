#!/usr/bin/env bash
# ============================================================
# CLEAN START — Reset bot state for fresh paper trading
# ============================================================
# Run on server AFTER stopping the bot:
#   tmux send-keys -t bot C-c
#   bash clean_start.sh
#   python3 live_multi_bot.py
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  CLEAN START — Trading Bot Reset"
echo "========================================"
echo ""

# Safety: refuse to run if bot is still alive
if pgrep -f "live_multi_bot" > /dev/null 2>&1; then
    echo "ERROR: live_multi_bot.py is still running!"
    echo "Stop it first: tmux send-keys -t bot C-c"
    exit 1
fi

echo "This will DELETE:"
echo "  - trade_journal/journal.db  (all trade history)"
echo "  - models/rl_*.pkl           (all ML models)"
echo "  - live_results/*            (bot state, equity curves, logs)"
echo "  - teacher_feedback.jsonl    (teacher grades)"
echo ""
echo "Backtest data and config are NOT touched."
echo ""
read -p "Continue? [y/N] " confirm
if [[ "${confirm,,}" != "y" ]]; then
    echo "Aborted."
    exit 0
fi

echo ""

# 1. Journal
if [[ -f trade_journal/journal.db ]]; then
    rm trade_journal/journal.db
    echo "[x] Deleted trade_journal/journal.db"
else
    echo "[ ] trade_journal/journal.db not found (skip)"
fi

# 2. ML Models (RL entry filter, exit classifier, BE manager, brain)
count=0
for f in models/rl_*.pkl models/dqn_*.zip; do
    if [[ -f "$f" ]]; then
        rm "$f"
        count=$((count + 1))
    fi
done
echo "[x] Deleted $count model files"

# 3. Bot state files (live_results/)
if [[ -d live_results ]]; then
    count=$(find live_results -type f | wc -l | tr -d ' ')
    rm -rf live_results/*
    echo "[x] Cleared live_results/ ($count files)"
else
    echo "[ ] live_results/ not found (skip)"
fi

# 4. Teacher feedback
if [[ -f teacher_feedback.jsonl ]]; then
    rm teacher_feedback.jsonl
    echo "[x] Deleted teacher_feedback.jsonl"
fi

# 5. Near-miss log
if [[ -f near_miss_log.jsonl ]]; then
    rm near_miss_log.jsonl
    echo "[x] Deleted near_miss_log.jsonl"
fi

echo ""
echo "========================================"
echo "  DONE — Clean state ready"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Reset balances on exchanges (Binance Testnet, OANDA Practice, Alpaca Paper)"
echo "  2. Start bot:  python3 live_multi_bot.py"
echo "  3. Monitor:    open dashboard or tail -f live_results/bot_001.log"
echo ""
