# A+B+C Architecture — Brainstorm Notes

**Purpose**: Pre-work notes for the ralplan session the user wants to do AFTER the train/inference mismatch fix is deployed. NOT a plan. NOT final. Just structured thinking to speed up the ralplan session.

---

## Recap: the user's original idea

> "retrain nach jedem geschlossenen trade"

The user wanted single-sample online learning after each closed trade (or each rejected-signal outcome). I pushed back on this as risky (statistical noise, catastrophic forgetting) and proposed **A + B + C**:

- **A — Counterfactual Continuous Learner**: include rejected signal outcomes in training samples, trigger batch retrains every N new samples
- **B — Exploration Budget (Multi-Armed Bandit)**: accept 5-10% of borderline rejections with reduced risk
- **C — Stability Guards**: shadow-training, rollback window, anomaly detection

---

## A — Counterfactual Continuous Learner (CCL)

### Current state
`continuous_learner.py` has:
- 1h check interval
- Triggers retrain when ≥50 new live trades accumulate
- Full XGBoost retrain (not warm-start) on backtest + live data
- Atomic write with `_prev.pkl` backup
- Gate check: validation PF ≥ 0.9× backtest PF before accepting new model

**Problem**: With sniper rate (~1 trade/week), 50 trades = 50 weeks of data. Way too slow to adapt.

### Proposed change
Extend the sample source:
- Count **rejected signals with simulated outcomes** as training samples too
- Simulated outcomes come from the same logic as `generate_rl_data.py`: forward simulation with SL/TP constraints

### Implementation sketch

```python
# In continuous_learner.py

def _load_rejected_signal_outcomes(db_path, since_ts):
    """Load rejected signals from trade_journal's 'rejections' table.
    Each row has: symbol, ts, features, direction, sl, tp, simulated_outcome.
    Triggered once a signal's simulated forward window has elapsed."""
    ...

def _combine_real_and_counterfactual(real_trades, rejected_outcomes):
    """Merge real and counterfactual samples. Real trades get weight=1.0,
    counterfactuals get weight=0.5 (less trustworthy due to no slippage/latency)."""
    ...

def retrain_if_ready(config):
    min_samples = config.get("min_samples_for_retrain", 20)  # lowered from 50
    real = _load_real_trades_since(last_retrain_ts)
    rejected = _load_rejected_signal_outcomes(db, last_retrain_ts)
    total = len(real) + len(rejected)
    if total < min_samples:
        return False
    ...
```

### Key design decisions
1. **Sample weight**: real=1.0, counterfactual=0.5. Protects against over-fitting to simulated outcomes.
2. **Min-sample threshold**: drop from 50 → 20 (since we now have many more samples).
3. **Counterfactual outcome TTL**: a rejected signal is tracked for `MAX_FORWARD_BARS` (576 bars = 48h). After that, the simulated outcome is "frozen" and added to training.
4. **DB schema change**: add `rejected_signals` table with `features (JSON)`, `direction`, `planned_sl/tp`, `created_at`, `outcome_at`, `outcome (win/loss/be/timeout)`, `rr_realized`.

### Risks
- **Simulation ≠ reality**: slippage, spreads, reject rates during rate limits, partial fills — none captured. Mitigation: reduced sample weight + deploy behind B (exploration budget) to gather REAL execution data too.
- **Simulation look-ahead**: if we simulate forward from a rejection, we use future data at training time. That's fine IF the simulation matches training-time labels (not live inference).
- **Feedback loop**: if the model accepts a signal and it loses, next retrain sees that loss. If it rejects and simulation says it would have won, next retrain sees that win. The model learns from both accepts and rejects — more robust.

---

## B — Exploration Budget (Multi-Armed Bandit)

### Problem
When the filter rejects everything, it never sees any REAL execution data. Exploration forces some rejections through as "explorer trades" to generate real feedback.

### Design

```
On every XGB REJECT with conf in [0.45, 0.55]:
  - roll a random number
  - if random < EXPLORATION_PCT (e.g. 0.10):
    - accept the signal as "explorer"
    - reduce risk to 0.25% (half normal)
    - tag the trade with explorer=True
  - else:
    - normal reject
```

### Dynamic exploration budget
If the filter has been rejecting everything for ≥24h, temporarily raise EXPLORATION_PCT from 10% to 25%. This breaks "filter lock".

### Safety rails
- Max 1 explorer trade per symbol per day
- Max 3 concurrent explorer trades total
- Explorer trades bypass daily-loss circuit breaker only within 0.25% each (so max cumulative explorer loss = 0.75% before circuit breaker kicks in)
- Mark explorer trades clearly in journal for post-hoc analysis

### Implementation location
`live_multi_bot.py` — `predict_entry()` caller, after `xgb_take = False`:

```python
if not xgb_take and rl_confidence >= 0.45:
    if self._should_explore(rl_confidence):
        self.logger.info("EXPLORER trade %s conf=%.3f — overriding reject", symbol, rl_confidence)
        xgb_take = True
        self._explorer_risk_scale = 0.5  # half risk
```

### Risks
- **Explorer loss pile-up**: 10% of 50 rejections = 5 explorer trades. If all lose, total drawdown = 5 × 0.25% = 1.25%. Manageable.
- **Exploration bias**: if exploration always picks borderline cases (conf ~0.5), we never test high-conf rejections. Counter: stratified exploration across conf ranges.

---

## C — Stability Guards

### C1: Shadow-training gate
Before replacing the production model, the new model must:
- Pass AUC ≥ old_model_auc - 0.02 on holdout
- Pass PF ≥ 0.9 × old_model_pf on holdout
- Have best_iteration ≥ 50 (avoid underfitting failures like Model A)
- Have no feature with zero variance (feature collapse)

### C2: Rollback window
Keep last 3 models in `models/history/`:
- `rl_entry_filter.pkl` — current
- `rl_entry_filter_prev.pkl` — yesterday
- `rl_entry_filter_prev2.pkl` — day before

If live DD exceeds -3% within 24h of deploying new model → auto-rollback to prev.

### C3: Anomaly detection on inference
After each retrain, compare prediction distribution on a **fixed test set** (e.g., 1000 cached samples):
- If mean conf changes by >0.15 vs previous model → alert
- If accept rate changes by >30% vs previous → require manual approval

### C4: Deployment canary
Deploy new model to ONE asset class first (e.g. commodities). Watch for 4-12h. Then roll out to others. Stops an architecture-wide failure.

---

## Proposed implementation order

1. **Deploy the optfix** (current fix) first — baseline with working features
2. **Monitor 24h** — confirm conf > 0.5 sustained, some trades going through
3. **Implement C first** (stability guards) — safety before experimentation
4. **Implement A** (counterfactual learner) — needs DB schema migration
5. **Implement B** (exploration budget) — LAST, since it touches live risk directly

Rationale: C is pure safety net, add it first. A is data plumbing, low risk. B changes live behavior, highest risk — add only after A+C are proven.

---

## Open questions for ralplan session

1. **Counterfactual sample weight**: 0.5 is a guess. Tune via ablation study?
2. **Exploration budget %**: 10% baseline feels right. Dynamic adjustment rules?
3. **Rollback trigger threshold**: -3% DD in 24h or different criteria?
4. **Can we retire the old journal for the new one**, or coexist?
5. **How to handle schema version bumps when adding counterfactual fields**?

---

## NOT in scope for A+B+C

- Changing the base model architecture (XGBoost stays)
- Changing the alignment score formula (stays as current 9-component after optfix)
- Changing the SMC strategy (stays)
- Adding new feature types (stays at 42 features)

These are deliberately held back to limit blast radius.

---

**Status**: BRAINSTORM — not committed to. Refinement happens in ralplan session with Planner/Architect/Critic agents.
