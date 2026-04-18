# Post-Deployment Verification Checklist

Run immediately after `scripts/deploy_optfix.sh --i-know-what-im-doing`.

## Phase 1: 0-5 min (startup)

- [ ] **Service is up**: `ssh server 'systemctl is-active trading-bot.service'` → `active`
- [ ] **No startup errors**: `ssh server 'journalctl -u trading-bot.service --since "5 min ago" | grep -i "error\|traceback" | head'` → empty
- [ ] **Adapters connected**: look for `Binance WS connected`, `OandaAdapter connected`, `Alpaca` in log
- [ ] **All 112 bots initialized**: `grep "Initialised bot_" /root/bot/paper_trading.log | tail -5`
- [ ] **Optimized SMC params loaded**: `grep "Loaded optimized SMC params" /root/bot/paper_trading.log | tail -1` → should show 322 symbol keys
- [ ] **XGB features validated**: `grep "XGB features validated" /root/bot/paper_trading.log | tail -1` → should say "42/42 exact match"

## Phase 2: 5-30 min (first decisions)

- [ ] **First heartbeat** with positions=1 (AUD_USD still open): `grep HEARTBEAT /root/bot/paper_trading.log | tail -3`
- [ ] **NEAR-MISS signals appearing**: `grep "NEAR-MISS ALIGNMENT" /root/bot/paper_trading.log | tail -5`
- [ ] **At least one XGB decision**: `grep -E "XGB (ACCEPT|REJECT)" /root/bot/paper_trading.log | tail -5`
- [ ] **Accept rate > 0** (not the old 0%): count in last 20 decisions
  ```bash
  ssh server 'grep -E "XGB ACCEPT|XGB REJECT" /root/bot/paper_trading.log | tail -20 | grep -c ACCEPT'
  ```
- [ ] **Confidence distribution** — look for conf > 0.5 (was <0.5 before)
  ```bash
  ssh server 'grep "XGB (ACCEPT|REJECT)" /root/bot/paper_trading.log | tail -20 | grep -oP "conf=[0-9.]+"'
  ```

## Phase 3: 30-120 min (stability)

- [ ] **AUD_USD #259 still managed correctly**: check that exit logic works
  ```bash
  ssh server 'cd /root/bot && sqlite3 trade_journal/journal.db "SELECT * FROM trades WHERE trade_id=259;"'
  ```
- [ ] **No bracket failures after new deploy**: `grep "Bracket order failed" /root/bot/paper_trading.log | tail -5` → no new entries
- [ ] **JPY pair orders succeed**: if any AUD_JPY signal passes XGB, it should NOT get `PRICE_PRECISION_EXCEEDED`
- [ ] **RL SL/TP adjustments working**: `grep "RL SL adjusted\|RL TP" /root/bot/paper_trading.log | tail -5`
- [ ] **Memory stable** (< 2 GB bot usage):
  ```bash
  ssh server 'systemctl status trading-bot.service | grep Memory'
  ```
- [ ] **No RUNTIME errors in last 30 min**:
  ```bash
  ssh server 'journalctl -u trading-bot.service --since "30 min ago" | grep -c ERROR'
  ```

## Phase 4: 2-24 hours (validation)

- [ ] **Accept rate vs reject rate**: Target ~10-30% accept (much higher than pre-fix 0-5%)
  ```bash
  ssh server 'cd /root/bot && sqlite3 trade_journal/journal.db "SELECT COUNT(*) FROM trades WHERE entry_time > \"2026-04-14T15:00\";"'
  ```
- [ ] **Conf distribution**: target mean ~0.6+, max ~0.8+, min ~0.4+
- [ ] **New trades opened** (expect 1-5 per day with sniper mode)
- [ ] **Any new trades profitable?** Look at closed ones' outcomes
- [ ] **No feature parity issues**: run `backtest/verify_feature_parity.py` on a fresh bar — should show mean_delta < 0.01

## Rollback triggers (auto-rollback conditions)

Execute `scripts/deploy_optfix.sh --rollback --i-know-what-im-doing` IF:
- [ ] Service crashes within 10 min of deploy
- [ ] Memory usage > 4 GB (was ~500 MB baseline)
- [ ] Accept rate suddenly drops to 0% again (regression)
- [ ] Multiple bracket order failures in first 30 min
- [ ] Catastrophic drawdown: -2% equity in first 60 min
- [ ] Any data corruption in trade journal

## After 24h: promote to "confirmed"

Once all checks pass for 24h, update `DEPLOYMENT_READINESS.md` with status `DEPLOYED` and record the deploy timestamp. Archive `DEPLOYMENT_READINESS.md` to `docs/deployments/optfix_YYYY-MM-DD.md`.

---

## Quick one-liner health check

```bash
ssh server 'systemctl is-active trading-bot.service && \
  echo "--- last 3 heartbeats ---" && \
  grep HEARTBEAT /root/bot/paper_trading.log | tail -3 && \
  echo "--- last 5 XGB decisions ---" && \
  grep -E "XGB (ACCEPT|REJECT)" /root/bot/paper_trading.log | tail -5 && \
  echo "--- errors last 30 min ---" && \
  grep ERROR /root/bot/paper_trading.log | tail -5'
```

Or run `scripts/status.sh` from local for full dashboard.
