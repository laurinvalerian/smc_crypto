# CLAUDE.md – SMC Crypto-Only AAA++ Trading Bot

## WICHTIG: Wo lebt welches Wissen?

- **CLAUDE.md** — Projektübersicht, Dateistruktur, Commands, Cross-Cutting Concerns
- **`.claude/agents/*.md`** — Domain-spezifisches Wissen (siehe Agent-Referenz unten)
- **`.omc/plans/crypto-only-refocus.md`** — Aktive Refactoring-Roadmap (RALPLAN-DR APPROVED)
- Bei Änderungen: CLAUDE.md UND den zuständigen Agent aktualisieren

## Projektübersicht

Crypto-Only Trading Bot basierend auf Smart Money Concepts (SMC/ICT) auf Binance USDT-M Futures. Sniper-Ansatz: nur AAA++ Setups (höchste Konfluenz), wenige Trades, maximale Profitabilität. Single-Asset-Class, kein Multi-Broker-Overhead, kein Tick-Volume-Hack.

**Refocus 2026-04-18**: Multi-Asset (Forex/Stocks/Commodities) entfernt. Begründung in `.omc/plans/crypto-only-refocus.md`.

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
| `filters/` | AAA++ Filter-Module (trend_strength, volume_liquidity, session_filter, zone_quality) |
| `exchanges/` | Binance USDT-M Futures Adapter (CCXT) |
| `risk/` | Circuit Breakers (Daily/Weekly/All-Time Loss, Heat) |
| `config/default_config.yaml` | Alle Parameter (Crypto-Only) |
| `utils/` | Crypto Data Downloader (CCXT/Binance) + Prefetch History |

### Signal-Flow (Kurzversion)

```
5m Candle → _prepare_signal() → Circuit Breaker → Volatility/Volume Gates
  → Discount/Premium Filter → Alignment Score (0.0-1.0)
  → Tier-Gate (alignment_threshold = 0.78) → Student/XGB Gate → Bracket Order
```

Vollständiger Signal-Flow: siehe `@agents/smc-strategist.md`. Phase 2.1 wird `core/alignment.py` als Single Source of Truth einführen.

### Tier-System (Threshold konsolidiert auf 0.78)

- **AAA++**: Score ≥ 0.88, RR ≥ 3.0, ALLE 11 Komponenten-Flags True → 1-1.5% Risk
- **AAA+**: Score ≥ 0.78, RR ≥ 2.0, 6 Kern-Flags True → 0.5-1% Risk
- Alles andere wird abgelehnt.

**Threshold-Konsolidierung 2026-04-18**: Vorher 3-fach Diskrepanz (Config 0.65 / Code 0.78 / CLAUDE.md 0.88). Jetzt: **0.78 in Config + Code als SSOT**. Tier-Schwellen 0.88/0.78 bleiben in `aaa_plus_plus.tiers` strukturell erhalten.

## Roadmap (Crypto-Only Refocus)

| Phase | Status | Tag |
|-------|--------|-----|
| Phase 0: Stabilisierung (Bug_006/007 Fix, Snapshots, Baseline) | ✅ FERTIG | `v1.0-multi-asset` |
| Phase 1: Asset-Class Strip | ✅ FERTIG | `v1.1-crypto-only-stripped` |
| Phase 2: Core Simplification (SSOT, ML-Friedhof, Teacher/Student-Decision, Retrain) | 🔄 LÄUFT (2.1 ✅, 2.2 ✅, 2.4 ⏳) | `v1.2-core-clean` (geplant), `v1.3-ssot-alignment` (2026-04-18) |
| Phase 3: Code Restructuring (live_multi_bot Aufspaltung) | ⏳ TODO | `v1.3-restructured` |
| Phase 4: ML-Konsolidierung + Funded-Compliance (Calendar-Day Fix) | ⏳ TODO | `v1.4-funded-ready` |
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
ALIGNMENT_THRESHOLD = 0.78       # Live-Wert (war fälschlich 0.88 dokumentiert)
MIN_RR = 3.0                     # AAA++ Trades
LEVERAGE = 5                     # Konservativ (Cap: 10x für Crypto)
RISK_PER_TRADE = 0.005           # 0.5% (konservativer Fallback)
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
bot/
├── .env / .env.example            # API Keys
├── check_downloads.sh / check_backtest.sh
├── live_multi_bot.py              # Haupt-Orchestrator (PaperBot + Runner) — Phase 3 spaltet auf
├── paper_grid.py                  # Multi-Variant A/B Testing (Crypto)
├── rl_brain_v2.py                 # XGBoost RLBrainSuite (Entry-Filter + BE-Manager aktiv)
├── train_student.py               # Student-Brain Training
├── continuous_learner.py          # Auto-Retrain (Phase 2.5: Cleanup)
├── drift_monitor.py               # KS/PSI Drift-Detection
├── trade_journal.py               # SQLite Trade-Logging (Bug_006 fixed)
├── live_teacher.py                # Live Teacher-Feedback
├── strategies/
│   └── smc_multi_style.py         # SMC/ICT Strategie (kausale Indikatoren V16)
├── filters/                       # AAA++ Filter-Module
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
│   └── circuit_breaker.py         # Daily/Weekly/All-Time DD (Phase 4: Calendar-Day Fix)
├── archive/
│   ├── multi_asset_legacy/        # OANDA/Alpaca-Adapter, Forex/Stocks-Downloader, ranker/, rl_brain.py
│   ├── models/                    # 10 deaktivierte/legacy .pkl
│   └── backtest_experimental/     # 16 Ad-hoc-Skripte
└── snapshots/                     # DB + Parquet-Snapshots pro Phase (gitignored)
```

## Agent-Referenz

| Domain | Agent-Datei | Zuständig für |
|--------|-------------|--------------|
| SMC/Strategie | `@agents/smc-strategist.md` | Signal-Flow, Komponenten, Tier-Flags, Filter, kausale Indikatoren (V16) |
| Backtesting | `@agents/backtester.md` | Walk-Forward, Optuna, Version-History, Anti-Overfitting, Bug_007 |
| Live Trading | `@agents/live-trader.md` | PaperBot, Runner, Paper Grid, Zombie Orders, RL Brain, Dashboard |
| Exchanges | `@agents/exchange-adapter.md` | Binance Adapter, Data Downloader, Prefetch |
| Risk | `@agents/risk-manager.md` | Circuit Breaker, Position Sizing, Funded Accounts (Calendar-Day Phase 4) |
| Performance | `@agents/perf-optimizer.md` | OOM Fixes, Parallelisierung, Memory, Server Constraints |
| Code Review | `@agents/trade-reviewer.md` | Lookahead Bias, Position Sizing, Timezone, Async Safety |
