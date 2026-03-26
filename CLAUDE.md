# CLAUDE.md – SMC Multi-Asset AAA++ Trading Bot

## WICHTIG: Wo lebt welches Wissen?

- **CLAUDE.md** — Projektübersicht, Dateistruktur, Commands, Cross-Cutting Concerns
- **`.claude/agents/*.md`** — Domain-spezifisches Wissen (siehe Agent-Referenz unten)
- Bei Änderungen: CLAUDE.md UND den zuständigen Agent aktualisieren

## Projektübersicht

Multi-Asset Trading Bot basierend auf Smart Money Concepts (SMC/ICT), der nur die absolut besten Trades (AAA++) über Crypto, Forex, Stocks und Commodities nimmt. Sniper-Ansatz: weniger Trades, höhere Qualität, maximale Profitabilität.

## Architektur

### Kernkomponenten

| Datei | Zweck |
|---|---|
| `live_multi_bot.py` | Multi-Asset Orchestrator (112 Bots: 30 Crypto + 28 Forex + 50 Stocks + 4 Commodities), PaperBot, Multi-TF Alignment, Order-Execution |
| `strategies/smc_multi_style.py` | SMC-Indikatoren (BOS, CHoCH, FVG, OB, Liquidity), kausale Implementierung (V16) |
| `rl_brain.py` | Zentrales PPO RL-Gehirn (24-dim Input, shared across alle Instrumente) |
| `paper_grid.py` | Multi-Variant A/B Testing (80 Varianten: 20 pro Asset-Klasse) |
| `filters/` | AAA++ Filter-Module (trend_strength, volume_liquidity, session_filter, zone_quality) |
| `exchanges/` | Exchange-Abstraktionsschicht (Binance, OANDA, Alpaca) |
| `ranker/` | Cross-Asset Opportunity Ranker (Scanner, Ranker, Allocator) |
| `risk/` | Circuit Breakers (Daily/Weekly Loss, Asset-Class Pause, Heat) |
| `config/default_config.yaml` | Alle konfigurierbaren Parameter (inkl. smc_profiles pro Asset-Klasse) |
| `utils/` | Data Downloaders (Crypto, Forex, Stocks) + Prefetch History |

### Signal-Flow (Kurzversion)

```
5m Candle → _prepare_signal() → Circuit Breaker → Volatility/Volume Gates
  → Discount/Premium Filter → 13-Komponenten Alignment Score (0.0-1.0)
  → Tier-Gate (AAA++ ≥0.88 / AAA+ ≥0.78) → RL Brain Gate → Bracket Order
```

Vollständiger Signal-Flow mit allen 13 Komponenten und Gewichten: siehe `@agents/smc-strategist.md`

### Tier-System

- **AAA++**: Score ≥ 0.88, RR ≥ 3.0, ALLE 11 Komponenten-Flags True → 1-1.5% Risk
- **AAA+**: Score ≥ 0.78, RR ≥ 2.0, 6 Kern-Flags True → 0.5-1% Risk
- Alles andere wird abgelehnt. Keine schwächeren Tiers.

Vollständige Flag-Liste: siehe `@agents/smc-strategist.md`

## Multi-Asset Roadmap

| Phase | Status |
|-------|--------|
| 1: AAA++ Filter | ✅ FERTIG |
| 2: Exchange-Abstraktionsschicht | ✅ FERTIG |
| 3: Multi-Asset Integration (OANDA, Alpaca) | ✅ FERTIG |
| 4: Cross-Asset Opportunity Ranker | ✅ FERTIG |
| 5: Circuit Breakers | ✅ FERTIG |
| Backtester V6→V16 (alle Bugs gefixt) | ✅ FERTIG |
| V16: Kausale SMC-Indikatoren | ✅ FERTIG |
| Live Bot Multi-Asset Refactoring | ✅ FERTIG |
| Paper Grid A/B Testing | ✅ FERTIG |

### Nächste Schritte

1. **V17 Grid-Optimierung**: ⏳ Grid von 3,024→216 Combos (nutzlose Params fixieren), be_ratchet≥1.5
2. **Paper Grid Varianten generieren**: ⏳ `--generate-paper-grid` (Stocks + Crypto zuerst)
3. **Paper Trading**: ⏳ 2 Wochen Demo — Stocks (hoch) + Crypto (mittel), Forex/Commodities pausiert
4. **Live → Funded**: 3× Funded Accounts geplant (erst nach Paper-Validierung)

### Paper-Trading Readiness (V16)

| Klasse | Ready? | Vertrauen | Begründung |
|--------|--------|-----------|------------|
| **Stocks** | JA | HOCH | PF 3.45, WR 40-70%, konservative Params (lev=1), sinkendes PF über Windows |
| **Crypto** | JA (vorsichtig) | MITTEL | PF 11.7+, konsistente Trades (25-31/W), niedrige DD |
| **Forex** | NEIN | NIEDRIG | 0 Evergreen Combos. SMC funktioniert nicht mit Tick-Volume |
| **Commodities** | TEILWEISE | MITTEL | PF 6.74, aber nur 4 Instrumente auf OANDA |

Detaillierte Backtest-Ergebnisse und Version-History (V6→V16): siehe `@agents/backtester.md`

## Exchanges

| Asset-Klasse | Exchange/Broker | Status |
|---|---|---|
| Crypto | Binance USDT-M Futures (CCXT) | ✅ Testnet aktiv |
| Forex | OANDA v20 API | ✅ 28 Pairs (`pip install v20`) |
| Stocks | Alpaca REST API | ✅ Top 50 (`pip install alpaca-py`) |
| Commodities | OANDA (XAU, XAG, WTI, BCO) | ✅ via OandaAdapter |

Details: siehe `@agents/exchange-adapter.md`

## Commands

```bash
# Live/Paper Trading
python3 live_multi_bot.py [--config config/default_config.yaml]

# Backtesting
python3 -m backtest.optuna_backtester                         # Global
python3 -m backtest.optuna_backtester --per-class             # Per Asset-Klasse
python3 -m backtest.optuna_backtester --generate-paper-grid   # Varianten generieren

# Daten herunterladen
python3 -m utils.data_downloader --workers 3                  # Crypto (Binance)
python3 -m utils.forex_data_downloader                        # Forex + Commodities (OANDA)
python3 -m utils.forex_data_downloader --commodities-only     # Nur Commodities
python3 -m utils.stock_data_downloader                        # US Stocks (Alpaca)
python3 -m utils.stock_data_downloader --symbols AAPL MSFT    # Einzelne Stocks
python3 -m utils.prefetch_history                             # Higher-TF Prefetch (vor Backtest!)

# Monitoring
bash check_downloads.sh                                        # Download-Fortschritt
bash check_backtest.sh                                         # Backtest-Fortschritt
watch -n 30 bash check_backtest.sh                             # Live-Monitor
tail -f backtest/results/backtest.log                          # Detaillierter Log
```

## Wichtige Konstanten

```
# Fallback-Parameter (wenn kein Evergreen aus backtest/results/{class}/evergreen_params.json)
ALIGNMENT_THRESHOLD = 0.88       # Sniper: nur AAA++ Trades
MIN_RR = 3.0                     # Nur Trades mit RR ≥ 3.0
LEVERAGE = 5                     # Konservativ
RISK_PER_TRADE = 0.005           # 0.5% (konservativer Fallback)
MAX_EQUITY_FOR_SIZING = 2x       # Equity Cap bei 2× Initial

# Circuit Breaker — Details in @agents/risk-manager.md
DAILY_LOSS = -3%  |  WEEKLY_LOSS = -5%  |  ALLTIME_DD = -8% (PERMANENT STOP)
MAX_PORTFOLIO_HEAT = 6%  |  MAX_RISK_PER_TRADE = 1.5%

# Leverage Caps (V14, reduziert)
Crypto 10x  |  Forex 20x  |  Stocks 4x  |  Commodities 10x

# RL Brain
BASE_OBS_DIM = 24  |  HIDDEN_DIM = 128  |  WARMUP_TRADES = 100
```

## Testing & Anti-Overfitting

- Walk-Forward: 3 Monate Train → 1 Monat OOS, 3 Windows
- 7-Gate OOS-Validierung: PF≥1.5, Trades≥20, Sharpe≥0.5, MC robust, DD>-10%, Stability<50%, Quality
- Cross-Window Evergreen: PF≥1.5 auf JEDEM Window, Ranking nach min(pf_real)
- Per-Asset-Class Params (verschiedene pro Klasse, gleiche über Zeitfenster)
- RL Brain: Nur Paper-Trading (nach 100 Warmup-Trades), NICHT im Backtest

Details: siehe `@agents/backtester.md`

## Environment

- **API Keys**: In `.env` via `python-dotenv` (nie committen). Template: `.env.example`
- **Broker-Accounts**: Binance Testnet (5K USDT), OANDA Practice (100K USD), Alpaca Paper (100K USD)
- **Daten**: `data/crypto/` (601 files), `data/forex/` (168), `data/stocks/` (250), `data/commodities/` (24) — alle ✅
- **Funded Account Ziel**: 3× 100K. DD-Limits: Daily -5%, All-Time -10%
- **Server**: 4 Cores, 8GB RAM, 4GB Swap

## Dateistruktur

```
bot/
├── .env / .env.example            # API Keys
├── check_downloads.sh / check_backtest.sh
├── live_multi_bot.py              # Haupt-Orchestrator (PaperBot + Runner)
├── paper_grid.py                  # Multi-Variant A/B Testing
├── rl_brain.py                    # PPO RL Brain (24-dim, shaped rewards)
├── strategies/
│   └── smc_multi_style.py         # SMC/ICT Strategie (kausale Indikatoren V16)
├── filters/                       # AAA++ Filter-Module
│   ├── trend_strength.py          # ADX, Momentum, TF Agreement
│   ├── volume_liquidity.py        # 3-Layer Volume Scoring
│   ├── session_filter.py          # Session-Awareness
│   └── zone_quality.py            # Zone Decay, Unmitigated, Formation
├── config/
│   └── default_config.yaml        # Alle Parameter (inkl. smc_profiles)
├── backtest/
│   └── optuna_backtester.py       # Walk-Forward Optimizer
├── utils/
│   ├── data_downloader.py         # Crypto OHLCV (CCXT/Binance)
│   ├── forex_data_downloader.py   # Forex+Commodities (OANDA v20)
│   ├── stock_data_downloader.py   # US Stocks (Alpaca)
│   └── prefetch_history.py        # Auto-prefetch higher TFs
├── exchanges/
│   ├── models.py                  # InstrumentMeta, OrderResult, PositionInfo, BalanceInfo
│   ├── base.py                    # Abstract ExchangeAdapter Interface
│   ├── binance_adapter.py         # Crypto
│   ├── oanda_adapter.py           # Forex + Commodities
│   └── alpaca_adapter.py          # US Stocks
├── ranker/
│   ├── universe_scanner.py        # Scannt ~200 Instrumente
│   ├── opportunity_ranker.py      # Z-Score Ranking
│   └── capital_allocator.py       # Position Limits + Korrelation
└── risk/
    └── circuit_breaker.py         # Daily/Weekly/Class/All-Time DD Limits
```

## Agent-Referenz

| Domain | Agent-Datei | Zuständig für |
|--------|-------------|--------------|
| SMC/Strategie | `@agents/smc-strategist.md` | Signal-Flow, 13 Komponenten, Tier-Flags, Filter, kausale Indikatoren (V16), Forex-Fixes, smc_profiles |
| Backtesting | `@agents/backtester.md` | Walk-Forward, Optuna, Bruteforce, Version-History (V6→V16), alle Ergebnisse, Anti-Overfitting |
| Live Trading | `@agents/live-trader.md` | PaperBot, Runner, Paper Grid, Zombie Orders, RL Brain, Dashboard, Multi-Exchange |
| Exchanges | `@agents/exchange-adapter.md` | Binance/OANDA/Alpaca Adapter, Data Downloaders, Broker-Setup, Prefetch |
| Risk | `@agents/risk-manager.md` | Circuit Breaker, Capital Allocator, Ranker Pipeline, Position Sizing, Funded Accounts |
| Performance | `@agents/perf-optimizer.md` | OOM Fixes, Parallelisierung, Memory, Server Constraints |
| Code Review | `@agents/trade-reviewer.md` | Lookahead Bias, Position Sizing, Timezone, Async Safety, Trading-Bug Checklists |
