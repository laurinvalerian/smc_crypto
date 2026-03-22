#!/bin/bash
# Backtest status checker
# Usage: bash check_backtest.sh
#    or: watch -n 30 bash check_backtest.sh

RESULTS="/root/bot/backtest/results"
LOG="$RESULTS/backtest.log"
STDOUT="$RESULTS/backtest_stdout.log"

echo "═══════════════════════════════════════════"
echo "  Backtest Status ($(date '+%H:%M:%S'))"
echo "═══════════════════════════════════════════"

BT_PID=$(pgrep -f "optuna_backtester" 2>/dev/null | head -1)
if [ -n "$BT_PID" ]; then
    echo "  Process: Running (PID $BT_PID)"
    ps -p "$BT_PID" -o %cpu,%mem,etime --no-headers 2>/dev/null | \
        awk '{printf "  CPU: %s%%  MEM: %s%%  Elapsed: %s\n", $1, $2, $3}'
    # Show worker CPUs
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
fi

# Trial progress from stdout
if [ -f "$STDOUT" ]; then
    TRIAL_PROGRESS=$(tr '\r' '\n' < "$STDOUT" | grep -oP '\d+/\d+' | tail -1)
    if [ -n "$TRIAL_PROGRESS" ]; then
        echo "  Current trial: $TRIAL_PROGRESS"
    fi
fi
echo ""

# Result files
echo "  Result Files:"
for f in backtest_stats.json wfo_summary.csv validation_results.json; do
    if [ -f "$RESULTS/$f" ]; then
        echo "    ✅ $f"
    else
        echo "    ⏳ $f"
    fi
done
TRIAL_N=$(ls "$RESULTS"/trades_window*.csv 2>/dev/null | wc -l)
OOS_N=$(ls "$RESULTS"/oos_trades_w*.csv 2>/dev/null | wc -l)
MC_N=$(ls "$RESULTS"/monte_carlo_w*.json 2>/dev/null | wc -l)
echo "    Trials: $TRIAL_N | OOS: $OOS_N | MC: $MC_N"

# Final summary if done
if [ -f "$RESULTS/backtest_stats.json" ]; then
    echo ""
    echo "  ══ FINAL SUMMARY ══"
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
