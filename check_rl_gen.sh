#!/bin/bash
# RL Data Generation Progress Tracker
# Usage: watch -n 30 bash check_rl_gen.sh

LOG="data/rl_training/generation_v3.log"
if [ ! -f "$LOG" ]; then echo "No log file found"; exit 1; fi

# Total work items per class
CRYPTO_TOTAL=360   # 30 symbols x 12 windows
FOREX_TOTAL=336    # 28 symbols x 12 windows
STOCKS_TOTAL=600   # 50 symbols x 12 windows
COMMOD_TOTAL=48    # 4 symbols x 12 windows
GRAND_TOTAL=$((CRYPTO_TOTAL + FOREX_TOTAL + STOCKS_TOTAL + COMMOD_TOTAL))

# Detect which class is currently running
current_class=$(grep "GENERATING RL DATA:" "$LOG" | tail -1 | sed 's/.*GENERATING RL DATA: //' | sed 's/ .*//')
classes_done=$(grep "SUMMARY" "$LOG" | wc -l)

# Count completed items from Progress lines
last_progress=$(grep "Progress:" "$LOG" | tail -1)

# Calculate completed items across all finished classes
done_items=0
for cls in CRYPTO FOREX STOCKS COMMODITIES; do
    cls_lower=$(echo "$cls" | tr '[:upper:]' '[:lower:]')
    if grep -q "═══ ${cls_lower} SUMMARY" "$LOG" 2>/dev/null || grep -q "═══ ${cls} SUMMARY" "$LOG" 2>/dev/null; then
        case $cls in
            CRYPTO) done_items=$((done_items + CRYPTO_TOTAL)) ;;
            FOREX)  done_items=$((done_items + FOREX_TOTAL)) ;;
            STOCKS) done_items=$((done_items + STOCKS_TOTAL)) ;;
            COMMODITIES) done_items=$((done_items + COMMOD_TOTAL)) ;;
        esac
    fi
done

# Add current class progress
if [ -n "$last_progress" ]; then
    current_done=$(echo "$last_progress" | grep -oP '\d+(?=/)')
    if [ -n "$current_done" ]; then
        done_items=$((done_items + current_done))
    fi
fi

pct=$((done_items * 100 / GRAND_TOTAL))

# Progress bar (50 chars wide)
filled=$((pct / 2))
empty=$((50 - filled))
bar=$(printf "%${filled}s" | tr ' ' '#')
bar="${bar}$(printf "%${empty}s" | tr ' ' '-')"

# Time estimation
start_time=$(head -1 "$LOG" | grep -oP '\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}')
if [ -n "$start_time" ] && [ $done_items -gt 0 ]; then
    start_epoch=$(date -d "$start_time" +%s 2>/dev/null)
    now_epoch=$(date +%s)
    elapsed=$((now_epoch - start_epoch))
    elapsed_min=$((elapsed / 60))
    rate=$(echo "scale=2; $done_items / $elapsed" | bc 2>/dev/null)
    remaining_items=$((GRAND_TOTAL - done_items))
    if [ -n "$rate" ] && [ "$rate" != "0" ]; then
        eta_sec=$(echo "scale=0; $remaining_items / $rate" | bc 2>/dev/null)
        eta_min=$((eta_sec / 60))
        eta_hr=$((eta_min / 60))
        eta_rem_min=$((eta_min % 60))
        eta_str="${eta_hr}h${eta_rem_min}m"
    else
        eta_str="calculating..."
    fi
else
    elapsed_min=0
    eta_str="calculating..."
fi

# PID check
pid=$(pgrep -f "generate_rl_data" 2>/dev/null | head -1)
if [ -n "$pid" ]; then
    status="RUNNING (PID $pid)"
else
    if grep -q "═══ TOTAL ═══" "$LOG" 2>/dev/null; then
        status="COMPLETED"
    else
        status="STOPPED/CRASHED"
    fi
fi

echo "═══════════════════════════════════════════════════════════"
echo " RL Data Generation Progress"
echo "═══════════════════════════════════════════════════════════"
echo " Status:   $status"
echo " Current:  $current_class"
echo " Progress: [$bar] ${pct}%"
echo "           ${done_items}/${GRAND_TOTAL} work items"
echo " Elapsed:  ${elapsed_min}m | ETA: ${eta_str}"
echo "═══════════════════════════════════════════════════════════"
echo " Classes:"
for cls in CRYPTO FOREX STOCKS COMMODITIES; do
    cls_lower=$(echo "$cls" | tr '[:upper:]' '[:lower:]')
    if grep -q "═══ ${cls_lower} SUMMARY\|═══ ${cls} SUMMARY" "$LOG" 2>/dev/null; then
        echo "   $cls  ✅ done"
    elif [ "$current_class" = "$cls" ] || [ "$current_class" = "$cls_lower" ]; then
        echo "   $cls  ⏳ in progress"
    else
        echo "   $cls  ⏸  waiting"
    fi
done
echo "═══════════════════════════════════════════════════════════"
