# CLAUDE.md – SMC Crypto-Only Scalp-Day Hybrid Trading Bot

## WICHTIG: Wo lebt welches Wissen?

- **CLAUDE.md** — Projektübersicht, Dateistruktur, Commands, Cross-Cutting Concerns
- **`.claude/agents/*.md`** — Domain-spezifisches Wissen (siehe Agent-Referenz unten)
- **`.omc/plans/crypto-only-refocus.md`** — Aktive Refactoring-Roadmap (RALPLAN-DR APPROVED)
- Bei Änderungen: CLAUDE.md UND den zuständigen Agent aktualisieren

## Projektübersicht

Crypto-Only Trading Bot basierend auf Smart Money Concepts (SMC/ICT) auf Binance USDT-M Futures. **Scalp-Day Hybrid** (5m Entry, 4h max Hold): kurzer RL-Feedback-Loop, SMC-Strukturzyklen passen, Commission-robust. Single-Asset-Class, kein Multi-Broker-Overhead.

**Refocus 2026-04-18**: Multi-Asset (Forex/Stocks/Commodities) entfernt. Begründung in `.omc/plans/crypto-only-refocus.md`.

**Scalp-Day Decision 2026-04-19**: AAA++/AAA+ Tier-System komplett entfernt. Einziger Gate = `alignment_score ≥ ALIGNMENT_THRESHOLD (0.78)`. Risk-Sizing linear skaliert über `core/sizing.py::compute_risk_fraction` (0.5% → 1.5% zwischen Threshold und Score 1.0). Student size-Head multipliziert live. `max_hold_bars: 96 → 48` (8h → 4h).

## Architektur

### Kernkomponenten

| Datei | Zweck |
|---|---|
| `live_multi_bot.py` | Crypto-Only Orchestrator (30 Bots), PaperBot, Multi-TF Alignment, Order-Execution |
| `strategies/smc_multi_style.py` | SMC-Indikatoren (BOS, CHoCH, FVG, OB, Liquidity), kausale Implementierung (V16) |
| `rl_brain_v2.py` | XGBoost-Stack (Entry-Filter, BE-Manager) — Legacy-Modelle deaktiviert |
| `models/student_brain.py` | Teacher/Student-System (4 Heads: entry/sl/tp/size) |
| `teacher/teacher_v2.py` | Hindsight-Label-Generierung (MFE/MAE-basiert) |
| `train_student.py` | Training-Skript für Student-Brain |
| `paper_grid.py` | Multi-Variant A/B Testing (Crypto-Varianten) |
| `filters/` | SMC Quality Filter-Module (trend_strength, volume_liquidity, session_filter, zone_quality) |
| `core/sizing.py` | Confidence-based Risk-Sizing SSOT (linear 0.5%→1.5% zwischen threshold und 1.0) |
| `exchanges/` | Binance USDT-M Futures Adapter (CCXT) |
| `risk/` | Circuit Breakers (Daily/Weekly/All-Time Loss, Heat) |
| `config/default_config.yaml` | Alle Parameter (Crypto-Only) |
| `utils/` | Crypto Data Downloader (CCXT/Binance) + Prefetch History |

### Signal-Flow (Kurzversion)

```
5m Candle → _prepare_signal() → Circuit Breaker → Volatility/Volume Gates
  → Discount/Premium Filter → Alignment Score (0.0-1.0)
  → alignment ≥ 0.78? → Student/XGB Gate → Confidence-Sizing → Bracket Order
```

Vollständiger Signal-Flow: siehe `@agents/smc-strategist.md`. Phase 2.1 ✅ hat `core/alignment.py` als Single Source of Truth etabliert (Tag `v1.3-ssot-alignment`).

### Gate + Sizing (no tiers, 2026-04-19)

- **Einziger Entry-Gate**: `alignment_score ≥ 0.78` + `rr ≥ 2.0`
- **Risk-Sizing linear**: `core/sizing.py::compute_risk_fraction(score)`
  - At score 0.78 → 0.25% of equity (`DEFAULT_RISK_PER_TRADE`)
  - At score 1.00 → 1.00% of equity (`MAX_RISK_PER_TRADE`)
  - Linear interpolation dazwischen (2026-04-19: lowered from 0.5%/1.5% for Scalp-Day Hybrid more-trades regime)
- **Live-Multiplikator**: Student size-Head Prediction (clamped zu `size_floor/size_cap`)
- **Backtest**: `student_size_multiplier = 1.0`, sonst identisch

**History**: AAA++/AAA+ Tier-System (2026-04-18 konsolidiert auf 0.78/0.88 Schwellen) am 2026-04-19 komplett entfernt — redundant zu Student size-Head + linear scaling.

## Roadmap (Crypto-Only Refocus)

| Phase | Status | Tag |
|-------|--------|-----|
| Phase 0: Stabilisierung (Bug_006/007 Fix, Snapshots, Baseline) | ✅ FERTIG | `v1.0-multi-asset` |
| Phase 1: Asset-Class Strip | ✅ FERTIG | `v1.1-crypto-only-stripped` |
| Phase 2: Core Simplification (SSOT, ML-Friedhof, Teacher/Student-Decision, Retrain) | 🔄 LÄUFT (2.1 ✅, 2.2 ✅, 2.4 ⏳) | `v1.3-ssot-alignment` (2026-04-18) |
| Phase 3: Code Restructuring (live_multi_bot Aufspaltung) | 🔄 LÄUFT (Dashboard ✅, Block 0 record-close ownership ✅) | `v1.4-record-close-owned` (2026-04-19). Runner + PaperBot File-Split pending. |
| Phase 4: ML-Konsolidierung + Funded-Compliance (Calendar-Day Fix ✅) | 🔄 (CB Calendar-Day ✅) | `v1.3-cb-funded` |
| Phase 5: Validation (Walk-Forward + 4 Wochen Paper) | ⏳ TODO | `v1.5-validated` |
| Phase 6: Funded Account Go/No-Go | ⏳ Nach Phase 5 | — |

Vollständiger Plan: `.omc/plans/crypto-only-refocus.md`

## Exchange

| Asset-Klasse | Exchange/Broker | Status |
|---|---|---|
| Crypto | Binance USDT-M Futures (CCXT) | ✅ Testnet aktiv, 30 Symbols |

Details: siehe `@agents/exchange-adapter.md`

## Commands

```bash
# Live/Paper Trading
python3 live_multi_bot.py [--config config/default_config.yaml]

# Backtesting
python3 -m backtest.optuna_backtester                         # Walk-Forward Optuna
python3 -m backtest.optuna_backtester --generate-paper-grid   # Crypto-Varianten

# Daten herunterladen
python3 -m utils.data_downloader --workers 3                  # Crypto (Binance)
python3 -m utils.prefetch_history                             # Higher-TF Prefetch

# Training
python3 train_student.py                                      # Student-Brain Retrain

# Monitoring
bash check_downloads.sh                                        # Download-Fortschritt
bash check_backtest.sh                                         # Backtest-Fortschritt
tail -f backtest/results/backtest.log                          # Detaillierter Log
```

## Wichtige Konstanten (Crypto-Only)

```
ALIGNMENT_THRESHOLD = 0.78       # Einziger Entry-Gate (SSOT: core/constants.py)
MIN_RR = 2.0                     # Scalp-Day Hybrid Default (config.risk_reward.default)
SCALP_MAX_HOLD_BARS = 48         # 4h auf 5m (Scalp-Day Hybrid, 2026-04-19)
LEVERAGE = 5                     # Konservativ (Cap: 10x für Crypto)
DEFAULT_RISK_PER_TRADE = 0.0025  # 0.25% at alignment threshold (2026-04-19: lowered for more-trades regime)
MAX_RISK_PER_TRADE = 0.010       # 1.0% at alignment score 1.0 (linear scale; was 1.5% pre-2026-04-19)
MAX_EQUITY_FOR_SIZING = 2x       # Equity Cap bei 2× Initial

# Circuit Breaker — Phase 4: Calendar-Day-Fix für Funded-Compliance
DAILY_LOSS = -3%  |  WEEKLY_LOSS = -5%  |  ALLTIME_DD = -8% (PERMANENT STOP)
MAX_PORTFOLIO_HEAT = 6%  |  MAX_RISK_PER_TRADE = 1.5%

# Crypto-Only Konstanten
COMMISSION = 0.0004              # Binance Futures taker fee
SLIPPAGE = 0.0002                # Konservativ
LEVERAGE_CAP = 10x               # Crypto

# Student Brain (Phase 2.4 entscheidet final ob aktiv)
4 Heads: entry, sl, tp, size  |  AUC 0.74 (vs Legacy XGB 0.66)

# Legacy RLBrainSuite (deaktiviert in Config)
tp_optimizer, sl_adjuster, position_sizer, exit_classifier — alle disabled
Aktiv: rl_entry_filter (Legacy-Fallback), rl_be_manager
```

## Testing & Anti-Overfitting

- Walk-Forward: 3 Monate Train → 1 Monat OOS, 3 Windows
- 7-Gate OOS-Validierung: PF≥1.5, Trades≥20, Sharpe≥0.5, MC robust, DD>-10%, Stability<50%, Quality
- Cross-Window Evergreen: PF≥1.5 auf JEDEM Window, Ranking nach min(pf_real)
- **Crypto-Only Vorteil**: Mehr Samples pro Symbol, keine Tick-Volume-Hacks, 24/7 Daten
- Student-Brain: Trainiert auf Hindsight-Labels (MFE/MAE), validiert OOS
- Phase-Abort-Gates (siehe `.omc/plans/crypto-only-refocus.md` Section 7): Pro Phase definierte Toleranzen

Details: siehe `@agents/backtester.md`

## Environment

- **API Keys**: In `.env` via `python-dotenv` (nie committen). Template: `.env.example`
- **Broker-Account**: Binance Testnet (5K USDT)
- **Daten**: `data/crypto/` (Crypto OHLCV)
- **Funded Account Ziel**: 1× 100K initial. DD-Limits: Daily -5%, All-Time -10% (Calendar-Day in Phase 4)
- **Server**: 4 Cores, 8GB RAM, 4GB Swap

## Dateistruktur

```
bongus_rival/                      # Repo-Root
├── .env / .env.example            # API Keys
├── check_downloads.sh / check_backtest.sh
├── live_multi_bot.py              # Haupt-Orchestrator (PaperBot + Runner) — Phase 3 spaltet in bot/ auf
├── paper_grid.py                  # Multi-Variant A/B Testing (Crypto)
├── rl_brain_v2.py                 # XGBoost RLBrainSuite (Entry-Filter + BE-Manager aktiv)
├── train_student.py               # Student-Brain Training
├── continuous_learner.py          # Auto-Retrain (Phase 2.5: Cleanup)
├── drift_monitor.py               # KS/PSI Drift-Detection
├── trade_journal.py               # SQLite Trade-Logging (Bug_006 fixed)
├── live_teacher.py                # Live Teacher-Feedback
├── bot/                           # Phase 3 Extract-Ziel
│   ├── __init__.py
│   └── dashboard.py               # Rich TUI Dashboard (250 LOC, extrahiert 2026-04-18)
├── core/
│   ├── constants.py               # COMMISSION, SLIPPAGE, ALIGNMENT_THRESHOLD, SCALP_MAX_HOLD_BARS SSOT
│   ├── alignment.py               # compute_alignment_score + CORE_WEIGHTS_CRYPTO (Phase 2.1 SSOT)
│   └── sizing.py                  # compute_risk_fraction/amount (Scalp-Day Hybrid SSOT, 2026-04-19)
├── strategies/
│   └── smc_multi_style.py         # SMC/ICT Strategie (kausale Indikatoren V16)
├── filters/                       # SMC Quality Filter-Module
│   ├── trend_strength.py          # ADX, Momentum, TF Agreement
│   ├── volume_liquidity.py        # 3-Layer Volume Scoring
│   ├── session_filter.py          # Session-Awareness
│   └── zone_quality.py            # Zone Decay, Unmitigated, Formation
├── teacher/                       # Hindsight-Label-Generierung
│   ├── teacher_v2.py              # MFE/MAE-basierte optimal_* Labels
│   └── backfill_parquet.py        # Backfill Training-Parquets
├── models/
│   ├── student_brain.py           # 4-Head Student-Klasse
│   ├── student_{entry,sl,tp,size}.pkl  # Trainierte Heads
│   ├── rl_entry_filter.pkl        # Legacy Entry-Filter
│   ├── rl_be_manager.pkl          # BE-Manager
│   ├── rl_brain_v2_xgb.pkl        # Default RLBrainSuite-Pfad
│   └── symbol_ranks.json          # Per-Symbol Vol/Liq/Spread Ranks
├── config/
│   ├── default_config.yaml        # Crypto-Only Parameter
│   └── instrument_clusters.json   # Crypto-Cluster
├── backtest/
│   ├── optuna_backtester.py       # Walk-Forward Optimizer
│   └── generate_rl_data.py        # Training-Data Pipeline (Bug_007 fixed)
├── utils/
│   ├── data_downloader.py         # Crypto OHLCV (CCXT/Binance)
│   ├── indicators.py              # Technische Indikatoren
│   └── prefetch_history.py        # Higher-TF Prefetch
├── exchanges/
│   ├── models.py                  # InstrumentMeta, OrderResult, PositionInfo, BalanceInfo
│   ├── base.py                    # Abstract ExchangeAdapter
│   ├── binance_adapter.py         # Crypto Adapter
│   └── replay_adapter.py          # Backtest-Replay
├── risk/
│   └── circuit_breaker.py         # Daily/Weekly/All-Time DD (Calendar-Day ✅ Phase 4.5)
├── tests/                         # pytest suite (52 cases)
│   ├── test_alignment.py          # Bit-parity sweep (1536 cases) + Core SSOT
│   ├── test_bug_fixes.py          # Bug_006 / Bug_007 regression
│   ├── test_circuit_breaker.py    # Calendar-Day funded-compat
│   └── test_constants.py          # SSOT verification
├── archive/
│   ├── multi_asset_legacy/        # OANDA/Alpaca-Adapter, Forex/Stocks-Downloader, ranker/, rl_brain.py
│   ├── models/                    # 10 deaktivierte/legacy .pkl
│   └── backtest_experimental/     # 16 Ad-hoc-Skripte
└── snapshots/                     # DB + Parquet-Snapshots pro Phase (gitignored)
```

## Agent-Referenz

| Domain | Agent-Datei | Zuständig für |
|--------|-------------|--------------|
| SMC/Strategie | `@agents/smc-strategist.md` | Signal-Flow, Komponenten, Alignment-Gate, Filter, kausale Indikatoren (V16) |
| Backtesting | `@agents/backtester.md` | Walk-Forward, Optuna, Version-History, Anti-Overfitting, Bug_007 |
| Live Trading | `@agents/live-trader.md` | PaperBot, Runner, Paper Grid, Zombie Orders, RL Brain, Dashboard |
| Exchanges | `@agents/exchange-adapter.md` | Binance Adapter, Data Downloader, Prefetch |
| Risk | `@agents/risk-manager.md` | Circuit Breaker, Position Sizing, Funded Accounts (Calendar-Day Phase 4) |
| Performance | `@agents/perf-optimizer.md` | OOM Fixes, Parallelisierung, Memory, Server Constraints |
| Code Review | `@agents/trade-reviewer.md` | Lookahead Bias, Position Sizing, Timezone, Async Safety |
