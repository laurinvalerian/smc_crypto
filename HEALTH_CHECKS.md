# System Health Checks — SMC Trading Bot

> **Usage**: When user asks "läuft alles gut?", "is everything ok?", or similar status
> questions, work through this document top-to-bottom. Each check is a concrete command
> with an expected result and an escalation path. Update this file whenever the
> architecture changes or you deploy something that needs observation.

Last updated: 2026-04-17

---

## 🚦 Quick Health Check (≤30 seconds)

Run these first. If any fail, the bot might be in trouble and needs deep check.

```bash
# 1. Service active?
ssh server "systemctl is-active trading-bot.service"
# Expected: active

# 2. Last activity recent?
ssh server "tail -3 /root/bot/paper_trading.log"
# Expected: timestamp within last ~5 minutes (markets open) or last heartbeat

# 3. Heartbeat stats
ssh server "grep 'heartbeat' /root/bot/paper_trading.log | tail -1"
# Expected: sensible numbers (bots active, equity, positions)
```

## 📊 State Check (1-2 minutes)

### A. Service & process

```bash
ssh server "systemctl show trading-bot.service -p ActiveEnterTimestamp -p MainPID -p NRestarts"
```
- **ActiveEnterTimestamp**: should be the expected deploy/restart time
- **MainPID**: should exist
- **NRestarts**: should NOT be incrementing without reason — investigate if it jumps

### B. Open positions + equity

```bash
ssh server "tail -200 /root/bot/paper_trading.log | grep -E 'heartbeat|equity=|active=|Restored state.*active=[1-9]'"
```
- Should show running heartbeat every ~60s
- `active=1` or higher for bots holding positions
- AUD_USD #259 (opened 2026-04-14 09:25 UTC) may still be open — if closed, check outcome in trades table

### C. Trade journal

```bash
ssh server "sqlite3 /root/bot/trade_journal/journal.db 'SELECT COUNT(*) as total_trades, SUM(CASE WHEN outcome=\"win\" THEN 1 ELSE 0 END) as wins, SUM(CASE WHEN exit_time IS NULL THEN 1 ELSE 0 END) as open FROM trades'"
```
- Known baseline (2026-04-15): 4 total, 3 wins, 1 open (AUD_USD)
- Growth rate: ~2-3 closed trades/week

```bash
ssh server "sqlite3 /root/bot/trade_journal/journal.db 'SELECT trade_id, symbol, asset_class, outcome, rr_actual, pnl_pct FROM trades ORDER BY entry_time DESC LIMIT 5'"
```
- Inspect recent trades for sanity (wins are realistic, losses aren't blowouts)

### D. Continuous Learner status

```bash
ssh server "grep -E 'Continuous learner' /root/bot/paper_trading.log | tail -3"
```
- Expected: "Continuous learner started: N closed trades in journal" + "loop: interval=1h, min_trades=50"
- **Gate progress**: `N / 50` — when N hits 50, CL fires first retrain

```bash
ssh server "grep -E 'Continuous learner.*retrain' /root/bot/paper_trading.log | tail -5"
```
- Expected: no retrain entries yet (dormant). First real retrain expected ~16 weeks after 2026-04-15.

### E. Drift Monitor status

```bash
ssh server "cat /root/bot/data/drift_state.json | python3 -m json.tool | head -30"
```
- **alert_level**: should trend from CRITICAL → WARNING → INFO over days after 2026-04-15 (stale pre-fix data ages out)
- **critical_major**: `['alignment_score', 'adx_1h']` is the OPTFIX-class drift pattern — if this reappears after normalization, INVESTIGATE immediately
- **major_count**: expected to drop as rejected_signals queue replenishes with post-fix data

```bash
ssh server "grep -E 'DRIFT|drift_monitor' /root/bot/paper_trading.log | tail -5"
```
- Expected: hourly check log with `[DRIFT OK]` / `[DRIFT WARNING]` / `[DRIFT CRITICAL]`

### F. Focal counterfactuals accumulation (weekly-ish)

```bash
ssh server "cd /root/bot && .venv/bin/python3 -c 'from continuous_learner import collect_counterfactual_focal_wins; df = collect_counterfactual_focal_wins(\"trade_journal/journal.db\"); print(f\"missed wins accumulated: {len(df)}\")'"
```
- **Why this matters**: tracks the "correction signal" building up for the next retrain. Higher numbers = more learning potential when CL fires.
- Expected growth: ~5-15 missed wins per week once forex rejects mature to their full 5d/24h/6h horizons
- **If it stays at 0 for weeks**: either market is very favorable (rejections were correct) OR something is wrong with outcome tracking

### G. Recent XGB decisions

```bash
ssh server "tail -1000 /root/bot/paper_trading.log | grep -E 'XGB ACCEPT|XGB REJECT' | tail -10"
```
- **Post F-fix (after 2026-04-15 06:48 UTC)**: XGB conf should be 0.3-0.8 range when a signal ≥0.78 passes the gate. Pre-fix was 0.1-0.5 and broken.
- **Accept rate**: expected ~30-50% of signals passing gate — if 0% for a week, fix isn't working
- **If conf suddenly drops to 0.1-0.3 range again**: F fix may have regressed — investigate drift_monitor alerts

### H. OANDA API health

```bash
ssh server "tail -500 /root/bot/paper_trading.log | grep -E 'OANDA|PRICE_PRECISION|ENTRY REJECTED|ORDER_FAILED'"
```
- **Post-S1-fix (2026-04-15)**: no more `PRICE_PRECISION_EXCEEDED` rejections for JPY pairs
- **If you see `ENTRY REJECTED` without corresponding retry**: S2 fix is catching a real broker rejection — note the reason
- `[ORDER_SUCCESS]` without `id=None` (S2 fix verified)

---

## 🔍 Current Watchlist (temporary, remove once resolved)

### W1: First post-F-fix XGB decision (active 2026-04-15)

**Goal**: Confirm the `_training_style_alignment_score` + `_training_style_adx_1h` helpers actually change XGB confidence on live signals.

**How to check**:
```bash
ssh server "awk '/2026-04-15 06:48:/,0' /root/bot/paper_trading.log | grep -E 'XGB ACCEPT|XGB REJECT' | head -5"
```

**Expected**: First decision with conf ≥ 0.3 (vs pre-fix ~0.1-0.2 range). Max near-miss score so far was 0.774 (just below 0.78 gate).

**Status 2026-04-15 08:15 UTC**: 0 XGB decisions yet — markt quiet, no signal ≥0.78.

**Resolve when**: First signal passes gate and shows sane conf (0.3-0.7). Then delete this watchlist item.

### W2: Drift monitor de-escalation (active 2026-04-15)

**Goal**: Confirm drift monitor will return to OK state as pre-fix rejected_signals age out.

**Current state (2026-04-15 07:17)**: `[DRIFT CRITICAL]` — 39 major drifts on stale pre-fix data.

**How to check**:
```bash
ssh server "cat /root/bot/data/drift_state.json | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d[\"alert_level\"], d[\"major_count\"], d[\"critical_major\"])'"
```

**Expected trajectory**:
- Day 1-2: CRITICAL — stale pre-fix data dominates
- Day 3-5: WARNING — mixed old/new
- Day 7+: OK if everything is working

**If STILL CRITICAL after 2 weeks**: something is wrong with F fix — investigate feature computation for alignment_score/adx_1h.

**Resolve when**: 7 days of OK in a row. Then delete this watchlist item.

### W3: Focal CF accumulation rate (active 2026-04-15)

**Goal**: Track how fast missed-win counterfactuals accumulate. Important for estimating when CL retrain will be meaningful.

**Current state**: 0 missed wins (all 49 signals are timeouts — market hasn't produced clear TP-hitting rejections).

**How to check**: See check F above.

**Expected**: If market is normal (mixed outcomes), should see 5-15/week. If 0 for 2+ weeks, market is unusually favorable OR outcome tracking has a bug.

**Resolve when**: 100+ missed wins accumulated. Then we have material for the first focal CF retrain.

### W4: SL adjuster — DISABLED 2026-04-17 (was ACTIVE 2026-04-15 → 2026-04-17)

**Status**: Disabled in config after backtest audit revealed structural design flaws:
- Model predicted `WIDEN` in 0 of 5.3M backtest samples across all 5 folds →
  de facto Tighten/Keep classifier.
- `_sl_eval` used `adjusted_rr[i] = actual_rr[i] / mult` mechanical scaling →
  reported +8 % PF was algebraic, not from better SL placement.
- Live evidence confirmed the bias: DOT 2026-04-17 SHORT tightened from 2.51 %
  to 1.76 % SL, got stopped at 1.336 while max adverse was only 1.52 % — the
  original SL would have held.

**Fixes applied 2026-04-17 (not yet re-enabled):**
- `derive_optimal_sl_label` (`rl_brain_v2.py:1314`): 5-bucket label set with WIDEN
  threshold lowered from 1.0R → 0.5R, added WIDEN_SIG at -1.0.
- `_sl_eval` (`rl_brain_v2.py:1349`): removed algebraic rescaling; outcomes only
  change when adjustment would have materially altered what happened.
- `predict_sl_adjustment` (`rl_brain_v2.py:2253`): matches new 5-bucket labels.
- `bug_008`: `live_multi_bot.py:2678-2690` and `trade_journal.py` now preserve
  the pre-adjustment SL as `sl_original`; adjusted value stored separately as
  `sl_used`. Journal column migration is idempotent.

**Re-enable gate**: 
1. Retrain SL on fixed labels (in progress 2026-04-17 ~14:21 UTC).
2. Backtest output must show > 0 WIDEN predictions in eval stats.
3. PF gain must be positive AND attributable to outcome changes, not algebra.
4. Shadow-mode deployment (log predictions, don't act) for ≥ 2 weeks.
5. Reactivate only after shadow log shows no systematic over-tightening.

### W5: TP optimizer — DISABLED 2026-04-17

**Status**: Disabled in config after backtest audit showed PF drops −0.03 to
−0.19 in 4/5 folds. Sharpe rose only because of WR increase (smaller wins, more
often), and `mfe_pred_corr` was 0.31–0.38 (too noisy for a regression target).

**How to check re-enable later**: retrain with PF gate (require > 0 improvement
in ≥ 4/5 folds) before flipping `config.rl_brain_v2.tp_optimizer.enabled` back
on.

### W6: Sample-weight rebalance 2026-04-17

**Goal**: Previous weights used `Grade D = 1.5` (same as A+), which boosted lucky
wins on bad setups as strongly as A-grade confirmed wins. New design: D base
= 0.5, with a 3× post-hoc boost for D+loss only so "learn from bad-grade
mistakes" intent is preserved without "chase lucky bad-grade wins."

**How to verify**: On the next retrain, inspect `sample_weight` distribution in
collected live data. D+win rows should have lower weights (≤ 2 × 0.5 × rr) and
D+loss rows should have higher weights (≥ 2 × 0.5 × 3 × outcome_mult).

**Resolve when**: 10+ D-grade trades have been retrained without regression.

---

## ⚠️ Red Flags — Immediate Investigation

If any of these appear, stop routine checks and dig in:

| Red flag | What it means | First action |
|---|---|---|
| NRestarts jumping | Bot is crashing | Check last 500 lines of log for ERROR/Traceback |
| `XGB conf < 0.3` for 10+ consecutive decisions | F fix regressed or pipeline drift | Run drift_monitor manually, compare state before/after |
| `ENTRY REJECTED` spike > 3/hour | Broker issues or S1 fix regressed | Check OANDA API status + per-symbol precision |
| Drift CRITICAL on a NEW feature (not alignment_score/adx_1h) | New mismatch bug — possibly from config change or data regen | Compare training parquet mtime vs live code paths |
| `[CRITICAL LOSS]` or equity drop > 2% in one day | Circuit breaker should have triggered | Verify risk/circuit_breaker.py is evaluating correctly |
| `max_db_size_mb` warning | Journal filling up | Check `sqlite3` file size + consider archiving |
| Memory %RSS > 5GB | continuous_learner skipping retrain | Check `psutil.Process(getpid()).memory_info()` — may indicate leak |
| Dashboard `/api/candles` returning empty | Bot not updating candle buffer | Check WebSocket connection logs for disconnects |

---

## 📁 Relevant files + scripts

- `live_multi_bot.py` — main runner + PaperBot class
- `continuous_learner.py` — retrain + focal CF logic
- `drift_monitor.py` — drift detection background job
- `rl_brain_v2.py` — XGBoost suite (entry/tp/sl/be models)
- `config/default_config.yaml` — all runtime params
- `data/drift_state.json` — latest drift monitor state (JSON)
- `trade_journal/journal.db` — trades + rejected_signals
- `.omc/research/insights/scripts/` — offline analysis scripts

---

## 🛠 Useful one-shot investigations

### Dump the latest N signals with their XGB conf + features
```bash
ssh server "sqlite3 /root/bot/trade_journal/journal.db \
  'SELECT timestamp, symbol, xgb_confidence, alignment_score FROM rejected_signals ORDER BY timestamp DESC LIMIT 10'"
```

### See which bot types are rejecting most
```bash
ssh server "tail -2000 /root/bot/paper_trading.log | grep 'XGB REJECT' | awk '{print \$6}' | sort | uniq -c | sort -rn | head"
```

### Check if focal CF will fire on next retrain (hypothetical)
```bash
ssh server "cd /root/bot && .venv/bin/python3 -c '
from continuous_learner import collect_counterfactual_focal_wins, collect_live_data
cf = collect_counterfactual_focal_wins(\"trade_journal/journal.db\")
live = collect_live_data(\"trade_journal/journal.db\", None, None, None)
print(f\"live trades: {len(live)} (need 50)\")
print(f\"focal missed wins: {len(cf)}\")
'"
```

### Tail a specific symbol's decisions
```bash
ssh server "tail -1000 /root/bot/paper_trading.log | grep 'AUD_JPY' | tail -20"
```

---

## 🗑 Deprecated / ignore

None currently. Delete items here once they're fully resolved + time has shown no regression.

---

**When updating this file**: 
- Add new watchlist items under "Current Watchlist" when you deploy something that needs observation
- Move resolved items to "Deprecated / ignore" with a date, then delete after 2 weeks
- Keep the "Quick Health Check" section stable — it's the 30-second entry point
