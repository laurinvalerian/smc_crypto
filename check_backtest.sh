#!/bin/bash
# Backtest status checker
# Usage: bash check_backtest.sh
#    or: watch -n 30 bash check_backtest.sh

RESULTS="/root/bot/backtest/results"
LOG="$RESULTS/backtest.log"
STDOUT="$RESULTS/backtest_v11.log"

echo "═══════════════════════════════════════════"
echo "  Backtest Status ($(date '+%H:%M:%S'))"
echo "═══════════════════════════════════════════"

BT_PID=$(pgrep -f "optuna_backtester" 2>/dev/null | head -1)
if [ -n "$BT_PID" ]; then
    echo "  Process: Running (PID $BT_PID)"
    ps -p "$BT_PID" -o %cpu,%mem,etime --no-headers 2>/dev/null | \
        awk '{printf "  CPU: %s%%  MEM: %s%%  Elapsed: %s\n", $1, $2, $3}'
    WORKERS=$(ps aux | grep "loky.*popen" | grep -v grep | wc -l)
    echo "  Workers: $WORKERS active"
else
    echo "  Process: NOT RUNNING"
fi
echo ""

# Window progress from log
if [ -f "$LOG" ]; then
    TOTAL_W=$(grep -c "^.*Window.*Train" "$LOG" 2>/dev/null)
    DONE_W=$(grep -c "Window.*OOS:" "$LOG" 2>/dev/null)
    echo "  Windows: ${DONE_W:-0} / ${TOTAL_W:-?} completed"

    # Show completed window results
    grep "Window.*OOS:" "$LOG" 2>/dev/null | while read -r line; do
        echo "    $(echo "$line" | sed 's/^.*Window/W/')"
    done
else
    echo "  Log: not yet created"
fi

# Trial progress from stdout (handles both \r progress bars and regular output)
if [ -f "$STDOUT" ]; then
    # Get latest trial progress line
    TRIAL_LINE=$(tr '\r' '\n' < "$STDOUT" | grep -E "Best trial.*\|" | tail -1)
    if [ -n "$TRIAL_LINE" ]; then
        TRIAL_PCT=$(echo "$TRIAL_LINE" | grep -oP '\d+%' | tail -1)
        TRIAL_NUM=$(echo "$TRIAL_LINE" | grep -oP '\d+/\d+' | tail -1)
        BEST_VAL=$(echo "$TRIAL_LINE" | grep -oP 'Best value: [0-9.]+' | tail -1)
        echo "  Current trial: ${TRIAL_NUM:-?} (${TRIAL_PCT:-?}) — ${BEST_VAL:-unknown}"
    fi

    # Window progress bar
    WF_LINE=$(tr '\r' '\n' < "$STDOUT" | grep -E "Walk-forward.*\|" | tail -1)
    if [ -n "$WF_LINE" ]; then
        WF_PCT=$(echo "$WF_LINE" | grep -oP '\d+%' | tail -1)
        WF_NUM=$(echo "$WF_LINE" | grep -oP '\d+/\d+' | tail -1)
        echo "  Walk-forward: ${WF_NUM:-?} (${WF_PCT:-?})"
    fi
fi
echo ""

# Signal cache files
SIG_COUNT=$(ls "$RESULTS"/signals_*.csv 2>/dev/null | wc -l)
echo "  Signal caches: $SIG_COUNT instruments"

# Result files
echo "  Result Files:"
for f in backtest_stats.json wfo_summary.csv validation_results.json; do
    if [ -f "$RESULTS/$f" ]; then
        echo "    [done] $f"
    else
        echo "    [wait] $f"
    fi
done
TRIAL_N=$(ls "$RESULTS"/top_params_w*.csv 2>/dev/null | wc -l)
echo "    Top params: $TRIAL_N windows"

# Final summary if done
if [ -f "$RESULTS/backtest_stats.json" ]; then
    echo ""
    echo "  == FINAL SUMMARY =="
    python3 -c "
import json
with open('$RESULTS/backtest_stats.json') as f:
    s = json.load(f)
print(f'  Mean PnL:     \${s[\"mean_pnl\"]:,.0f}')
print(f'  Mean PF:      {s[\"mean_profit_factor\"]:.2f}')
print(f'  Mean Sharpe:  {s[\"mean_sharpe\"]:.2f}')
print(f'  Mean WR:      {s[\"mean_winrate\"]*100:.1f}%')
print(f'  Worst DD:     {s[\"worst_drawdown\"]*100:.1f}%')
print(f'  Total Trades: {s[\"total_trades_all_windows\"]}')
print(f'  Validation:   {s[\"validation_passed\"]}/{s[\"validation_total\"]} passed')
" 2>/dev/null
fi
echo "═══════════════════════════════════════════"
