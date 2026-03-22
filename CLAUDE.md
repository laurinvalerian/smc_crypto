# CLAUDE.md – SMC Multi-Asset AAA++ Trading Bot

## WICHTIG: CLAUDE.md immer aktuell halten!

**Diese Datei MUSS nach jeder abgeschlossenen Phase, jedem neuen Modul, und jeder Architekturentscheidung sofort aktualisiert werden.** Da der Kontext regelmässig komprimiert wird, ist CLAUDE.md die einzige zuverlässige Quelle für den aktuellen Projektstand. Ohne aktuelle CLAUDE.md gehen kritische Details bei der Komprimierung verloren. Bei jeder Änderung: Dateistruktur, Roadmap-Status, Architektur-Sections und Konstanten prüfen und updaten.

## Projektübersicht

Multi-Asset Trading Bot basierend auf Smart Money Concepts (SMC/ICT), der nur die absolut besten Trades (AAA++) über Crypto, Forex, Stocks und Commodities nimmt. Sniper-Ansatz: weniger Trades, höhere Qualität, maximale Profitabilität.

## Architektur

### Kernkomponenten

| Datei | Zweck |
|---|---|
| `live_multi_bot.py` | Haupt-Orchestrator, PaperBot-Klasse, Multi-TF Alignment, Tier-Klassifizierung, Order-Execution |
| `strategies/smc_multi_style.py` | SMC-Indikatoren (BOS, CHoCH, FVG, OB, Liquidity), Entry-Zone-Erkennung |
| `rl_brain.py` | Zentrales PPO RL-Gehirn (24-dim Input, shared across alle Instrumente) |
| `filters/` | AAA++ Filter-Module (trend_strength, volume_liquidity, session_filter, zone_quality) |
| `exchanges/` | Exchange-Abstraktionsschicht (Binance, OANDA, Alpaca) |
| `ranker/` | Cross-Asset Opportunity Ranker (Scanner, Ranker, Allocator) |
| `risk/` | Circuit Breakers (Daily/Weekly Loss, Asset-Class Pause, Heat) |
| `config/default_config.yaml` | Alle konfigurierbaren Parameter |

### Signal-Flow (Top-Down)

```
5m Candle arrives → _prepare_signal()
  ├── Circuit Breaker Check (daily/weekly loss, asset-class pause, heat)
  ├── Volatility Gate (Daily ATR ≥ 0.8%, 5m ATR ≥ 0.15%)
  ├── Volume Pre-Check (≥ 0.5x 20-bar avg)
  ├── _multi_tf_alignment_score() → 13-Komponenten-Score (0.0-1.0)
  │   ├── 1D: Daily Bias via BOS/CHoCH (0.10)
  │   ├── 4H: Structure + POI (0.08 + 0.08)
  │   ├── 1H: Structure + CHoCH (0.08 + 0.06)
  │   ├── 15m: Entry Zone × Zone-Quality (0.12)
  │   ├── 5m: Precision Trigger BOS/CHoCH (0.10)
  │   ├── Volume: 3-Layer Score (0.08)
  │   ├── ADX: Trend Strength 1H (0.08)
  │   ├── Session: Optimality Score (0.06)
  │   ├── Momentum: RSI+MACD Confluence (0.06)
  │   ├── TF Agreement: EMA20/50 auf 4 TFs (0.05)
  │   └── Zone Freshness: Decay-Faktor (0.05)
  ├── Score < 0.65 → REJECT
  ├── SL/TP Berechnung + Style-Klassifizierung (scalp/day/swing)
  ├── Tier-Klassifizierung:
  │   ├── AAA++ (score ≥ 0.88, RR ≥ 5.0, ALLE Komponenten True) → 1-2% Risk
  │   └── AAA+  (score ≥ 0.78, RR ≥ 4.0, Kern-Komponenten True) → 0.5-1% Risk
  ├── RL Brain Gate (nach 100 Warmup-Trades)
  └── Bracket Order Execution (Market + SL + TP)
```

### Tier-System (nur AAA++ und AAA+)

**AAA++ (Sniper)** — Score ≥ 0.88, RR ≥ 5.0, erfordert:
- `bias_strong` (BOS/CHoCH-bestätigter Daily Bias)
- `h4_confirms` + `h4_poi` (4H Struktur + aktiver OB/FVG)
- `h1_confirms` + `h1_choch` (1H Struktur + CHoCH)
- `entry_zone` (15m FVG/OB)
- `precision_trigger` (5m BOS/CHoCH)
- `volume_ok` (3-Layer Volume Score ≥ 0.6)
- `adx_strong` (ADX > 25 auf 1H)
- `session_optimal` (Session Score ≥ 0.8)
- `zone_quality_ok` (Zone Quality ≥ 0.7)
- `momentum_confluent` (RSI + MACD aligned)
- `tf_agreement` ≥ 4 (alle 4 TFs EMA20/50 aligned)

**AAA+ (Fallback)** — Score ≥ 0.78, RR ≥ 4.0, erfordert:
- bias_strong, h4_confirms, h1_confirms, precision_trigger, volume_ok, adx_strong

**Keine A- oder SPEC-Tiers mehr.** Schwache Trades werden komplett abgelehnt.

### Filter-Module (`filters/`)

- **`trend_strength.py`** — ADX(14) mit Wilder-Smoothing, MACD-Histogram (EMA12/26/9), RSI(14), Multi-TF EMA20/50 Agreement
- **`volume_liquidity.py`** — Layer 1: Relative Volume vs 100-bar Avg (min 1.5x), Layer 2: Dollar-Volume Floor ($50K crypto, $100K forex/stocks), Layer 3: Volume-Profile auf 1H (high-vol entry + low-vol path to TP)
- **`session_filter.py`** — UTC Session-Scores: Crypto peak bei London/NY Open, Forex London+NY Overlap, Stocks nur Regular Hours, Commodities asset-spezifisch
- **`zone_quality.py`** — Exponentieller Decay (`exp(-0.15 × age)`), Unmitigated-Check (0/0.5/1.0), Zone-Size vs ATR (Sweet Spot 0.5-2.0 ATR), Formation-Quality (Body/Wick-Ratio), HTF-Overlap-Bonus

### RL Brain (`rl_brain.py`)

- **24-dim Observation** + 1 Coin/Instrument-ID = 25-dim Input
- **128-dim Hidden Layer** (2-Layer MLP mit Tanh)
- **Features 0-15**: Original (alignment, direction, ATR, EMAs, volume_ratio, returns, RSI, tier, style, RR, daily_ATR)
- **Features 16-23**: Neu (adx_normalized, session_score, zone_quality, volume_score, momentum_score, tf_agreement, spread, asset_class_id)
- **Shaped Reward**: `pnl × rr_quality_bonus × tier_bonus` + Quick-SL-Penalty
- **Architektur-Änderung erfordert Neutraining** — alte Checkpoints sind inkompatibel

### Exchange-Abstraktionsschicht (`exchanges/`)

**Interface (`ExchangeAdapter`)** — Jeder Adapter implementiert:
- `connect()` / `close()` — Lifecycle
- `load_markets()` → `dict[symbol, InstrumentMeta]` — Instrument-Metadaten cachen
- `fetch_ohlcv()` / `watch_ohlcv()` / `watch_ticker()` — Market Data (REST + WebSocket)
- `create_market_order()` / `create_stop_loss()` / `create_take_profit()` — Trading → `OrderResult`
- `fetch_balance()` → `BalanceInfo` / `fetch_positions()` → `list[PositionInfo]` — Account
- `set_leverage()` / `set_margin_mode()` / `fetch_max_leverage()` — Margin-Management
- `price_to_precision()` / `amount_to_precision()` — Precision-Helpers
- `is_market_open()` — Trading-Hours-Check

**`BinanceAdapter`** (Crypto) — Konkrete Implementierung:
- Wraps `ccxt.pro` (async, WebSocket) + `ccxt` sync (Startup-History)
- `raw` Property gibt direkten Zugriff auf ccxt.pro Objekt (für Migration)
- `load_markets()` parsed Binance-Filters (LOT_SIZE, MARKET_LOT_SIZE) für korrekte max_qty
- `fetch_max_leverage()` — 3-Methoden-Fallback (ccxt unified → fapiPrivateGetLeverageBracket → cached meta)

**`OandaAdapter`** (Forex + Commodities):
- OANDA v20 REST API (`pip install v20`)
- 28 Forex-Pairs (7 Majors + 21 Crosses) + 4 Commodities (XAU, XAG, WTI, BCO)
- `asyncio.to_thread()` für alle API-Calls (v20 ist sync)
- SL/TP über Trade-attached Orders (OANDA-spezifisch, kein Bracket)
- Spread-basiertes Pricing (keine Commission)
- Trading Hours: Forex 24/5 (So 22:00 UTC → Fr 22:00 UTC), Commodities ~23h/Tag
- Leverage: Forex max 30x, Commodities max 20x (reguliert)

**`AlpacaAdapter`** (US Stocks):
- Alpaca REST API (`pip install alpaca-py`)
- Top 50 US-Aktien nach Market Cap
- Fractional Shares Support
- Commission-free, max 4x Leverage (Reg T)
- Regular Trading Hours only (13:30-21:00 UTC, DST-aware)
- Position Sizing in Shares (nicht Lots)

### Cross-Asset Ranker (`ranker/`)

**Pipeline**: `UniverseScanner` → `OpportunityRanker` → `CapitalAllocator` → Execution

**`UniverseScanner`**:
- Hält Referenzen zu allen `ExchangeAdapter`-Instanzen
- Scannt alle Instrumente in Batches (10 parallel, rate-limit-aware)
- Pre-Filter: ATR ≥ 0.4%, Volume ≥ 0.5x 20-bar avg, Market open
- OHLCV-Cache mit 5min TTL (verhindert API-Overload)
- Lightweight Scores: EMA-Trend, RSI-Momentum, Volume-Ratio, Session
- Ergebnis: `UniverseState` mit allen `ScanResult`-Objekten

**`OpportunityRanker`**:
- Gruppiert Ergebnisse nach Asset-Klasse
- Z-Score-Normalisierung pro Komponente innerhalb jeder Klasse (Sigmoid-Mapping)
- Gewichteter Composite: alignment 35% + volume 20% + trend 15% + session 10% + zone_quality 10% + RR 10%
- 20% Bonus für Instrumente mit aktivem Trade-Signal
- Filtert auf min_opportunity_score (default 0.5)

**`CapitalAllocator`**:
- 5 sequentielle Checks: (1) Already in position, (2) Max total positions (5), (3) Max per class (3), (4) Portfolio heat (6%), (5) Correlation (Pearson > 0.7)
- Risk-Sizing: AAA++ 1-2%, AAA+ 0.5-1%, skaliert mit opportunity_score
- `PortfolioState` trackt positions, equity, return_series für Korrelation
- Ergebnis: `AllocationDecision` pro Opportunity (approved/rejected mit Grund)

**Datenmodelle (`models.py`)**:
- `InstrumentMeta` — symbol, exchange_symbol, asset_class, tick/lot_size, min/max_qty, max_leverage, trading_hours, commission_pct
- `OrderResult` — order_id, symbol, side, type, qty, price, status
- `PositionInfo` — symbol, side, qty, entry_price, unrealized_pnl, leverage
- `BalanceInfo` — currency, total, free, used

## Multi-Asset Roadmap

### Phase 1: AAA++ Filter (✅ FERTIG)
Neue Filter-Module, 13-Komponenten-Scoring, Tier-Umbau, RL-Brain-Erweiterung.

### Phase 2: Exchange-Abstraktionsschicht (✅ FERTIG)
- `exchanges/models.py` — `InstrumentMeta`, `OrderResult`, `PositionInfo`, `BalanceInfo` Dataclasses
- `exchanges/base.py` — Abstract `ExchangeAdapter` (alle Methoden: Market Data, Trading, Account, Leverage, Trading Hours)
- `exchanges/binance_adapter.py` — `BinanceAdapter` wraps ccxt.pro + ccxt sync, `raw` Property für schrittweise Migration
- `exchanges/__init__.py` — Package-Exports
- **Migration**: `live_multi_bot.py` nutzt `BinanceAdapter` über `adapter.raw` Property während der Übergangsphase. Schrittweise Umstellung auf Adapter-Methoden geplant.

### Phase 3: Multi-Asset Integration (✅ FERTIG)
- `exchanges/oanda_adapter.py` — Forex (28 Pairs) + 4 Commodities via OANDA v20 API
- `exchanges/alpaca_adapter.py` — Top 50 US Stocks via Alpaca REST API
- Asset-spezifische SMC-Profile bereits in `config/default_config.yaml` (Phase 1)
- Lazy Imports in `exchanges/__init__.py` (kein ImportError wenn v20/alpaca-py nicht installiert)

### Phase 4: Cross-Asset Opportunity Ranker (✅ FERTIG)
- `ranker/universe_scanner.py` — `UniverseScanner`: Scannt ~200 Instrumente über alle Adapter, OHLCV-Caching (5min TTL), ATR + Volume Pre-Filter, Batch-Scanning (10 parallel), `ScanResult` + `UniverseState` Dataclasses
- `ranker/opportunity_ranker.py` — `OpportunityRanker`: Z-Score-Normalisierung pro Asset-Klasse (Sigmoid-Mapping), gewichteter Composite-Score (alignment 35%, volume 20%, trend 15%, session 10%, zone_quality 10%, RR 10%), 20% Bonus für aktive Signale, min_opportunity_score Filter
- `ranker/capital_allocator.py` — `CapitalAllocator`: Max 5 Positionen total, max 3 pro Asset-Klasse, Pearson-Korrelation > 0.7 → reject, Portfolio Heat max 6%, Tier-basiertes Risk-Sizing (AAA++ 1-2%, AAA+ 0.5-1%), Leverage-Limits per Asset-Klasse (Crypto 20x, Forex 30x, Stocks 4x, Commodities 20x)
- `ranker/__init__.py` — Package-Exports
- **Architektur**: UniverseScanner → OpportunityRanker → CapitalAllocator → Trade Execution

### Phase 5: Circuit Breakers (✅ FERTIG)
- `risk/circuit_breaker.py` — `CircuitBreaker` + `CircuitBreakerState`
- Tagesverlust ≥ 3% → Stop ALL Trading für 24h
- Wochenverlust ≥ 5% → `size_reduction_factor = 0.5` (halbe Positionsgrößen)
- Asset-Class Drawdown ≥ 2% → Pause diese Klasse für 12h
- Portfolio Heat > 6% → Keine neuen Positionen
- `can_trade(asset_class)` → `(bool, reason)` Quick-Check vor jedem Entry
- `get_size_factor()` → Multiplikator für Position-Sizing (1.0 oder 0.5)
- Auto-Recovery: Pausen laufen automatisch ab, Size-Reduction hebt sich auf wenn Weekly-PnL erholt

### Integration in live_multi_bot.py (✅ FERTIG)
- `create_exchange()` erstellt `BinanceAdapter` intern, gibt `adapter.raw` zurück (backward-compat)
- `PaperBot.circuit_breaker` — Shared Circuit Breaker, Check in `_prepare_signal()` vor jedem Signal
- `LiveMultiBotRunner` initialisiert Circuit Breaker, Ranker, Allocator im Konstruktor
- Circuit Breaker PnL-Recording bei jedem Trade-Close in `_poll_positions()`
- Portfolio-Heat-Update im Dashboard-Loop (summiert risk_pct aller aktiven Trades)
- Ranker + Allocator sind initialisiert aber noch nicht trade-driving (bereit für Multi-Asset-Modus)

### Backtester (`backtest/optuna_backtester.py`) (✅ FERTIG)

Erweitert mit AAA++ Filter-Integration, Circuit Breaker Simulation und Anti-Overfitting-Gates:

**Kernfunktionen:**
- `classify_signal_tier()` — AAA++ / AAA+ / REJECTED basierend auf Score, RR, Komponenten-Flags
- `simulate_trades()` — Komplett umgeschrieben: AAA++ Tier-Gate, Circuit Breaker pro Trade, dynamisches Risk-Sizing (AAA++ 1-2%, AAA+ 0.5-1%), Size-Reduction-Faktor von CB
- `compute_metrics()` — Erweitert: avg_rr, trades pro Tier, pnl_per_trade, expectancy
- `monte_carlo_check()` — 1000x Trade-Reihenfolge shufflen, 95%-KI berechnen, robust wenn untere Grenze > 0
- `validate_oos_results()` — 4 Gates: PF≥1.5, min 100 Trades, Sharpe≥0.5, Monte Carlo robust
- `check_parameter_stability()` — ±10% Parameter-Perturbation, prüft ob PF-Änderung <50%

**Trade-Outcome-Modell:**
- Win-Probability: `alignment_score × 0.60 - RR_penalty` (jeder RR-Punkt >3.0 reduziert um 2%)
- Circuit Breaker simuliert Daily/Weekly Loss Limits pro Trade
- Seeded RNG für reproduzierbare Ergebnisse

**CLI-Flags:** `--monte-carlo`, `--stability-check`

### Nächste Schritte
1. **Datenquellen + Broker-Setup planen** (API Keys, Testnet/Paper, Account Balances)
2. **RL Brain Neutraining**: Alten Checkpoint löschen, 24-dim Architektur
3. **Paper Trading**: 2 Wochen Demo über alle Asset-Klassen
4. **Live**: Nur wenn Paper-Ergebnisse innerhalb 1σ der Backtests

## Testing & Anti-Overfitting

- **Walk-Forward-Validation Pflicht**: 3 Monate Train → 1 Monat Out-of-Sample
- **Out-of-Sample Profit Factor ≥ 1.5**
- **Minimum 100 Trades** im OOS für statistische Relevanz
- **Parameter-Stabilität**: ±10% Änderung darf Performance nicht kippen (Backtester: `check_parameter_stability()`)
- **Monte-Carlo**: Trade-Reihenfolge 1000x shufflen, 95%-KI muss profitabel sein (Backtester: `monte_carlo_check()`)
- **4-Gate OOS-Validierung**: PF≥1.5, Trades≥100, Sharpe≥0.5, Monte Carlo robust (Backtester: `validate_oos_results()`)
- **RL Pre-Training**: Nur mit Out-of-Sample Backtest-Trades, Curriculum-basiert
- **Paper-Trading**: Mindestens 2 Wochen vor Live, innerhalb 1 Std-Abweichung der Backtests

## Exchanges

| Asset-Klasse | Exchange/Broker | Status |
|---|---|---|
| Crypto | Binance USDT-M Futures (CCXT) | ✅ Adapter fertig, aktiv (Testnet) |
| Forex | OANDA v20 API | ✅ Adapter fertig (28 Pairs), braucht `pip install v20` |
| Stocks | Alpaca REST API | ✅ Adapter fertig (Top 50), braucht `pip install alpaca-py` |
| Commodities | OANDA (XAU, XAG, WTI, BCO) | ✅ Adapter fertig (via OandaAdapter) |

## Commands

```bash
# Live/Paper Trading starten
python3 live_multi_bot.py [--config config/default_config.yaml]

# Backtesting (Optuna Walk-Forward)
python3 backtest/optuna_backtester.py
```

## Wichtige Konstanten

```
ALIGNMENT_THRESHOLD = 0.65       # Minimum Score für Trade-Consideration
TIER_AAA_PLUS_PLUS: score ≥ 0.88, RR ≥ 5.0
TIER_AAA_PLUS:      score ≥ 0.78, RR ≥ 4.0
MIN_DAILY_ATR_PCT = 0.008        # 0.8% Volatility Floor
MIN_5M_ATR_PCT = 0.0015          # 0.15% per 5m Bar
MIN_SL_ATR_MULT = 2.5            # SL mindestens 2.5× ATR
BASE_OBS_DIM = 24                # RL Feature-Dimension
HIDDEN_DIM = 128                 # RL Network Hidden Layer
WARMUP_TRADES = 100              # Trades ohne RL Gate
```

## Dateistruktur

```
bot/
├── live_multi_bot.py              # Haupt-Orchestrator (PaperBot + Runner)
├── rl_brain.py                    # PPO RL Brain (24-dim, shaped rewards)
├── strategies/
│   └── smc_multi_style.py         # SMC/ICT Strategie (BOS, CHoCH, FVG, OB)
├── filters/                       # AAA++ Filter-Module
│   ├── __init__.py
│   ├── trend_strength.py          # ADX, Momentum, TF Agreement
│   ├── volume_liquidity.py        # 3-Layer Volume Scoring
│   ├── session_filter.py          # Session-Awareness
│   └── zone_quality.py            # Zone Decay, Unmitigated, Formation
├── config/
│   └── default_config.yaml        # Alle Parameter
├── backtest/
│   └── optuna_backtester.py       # Walk-Forward Optimizer
├── utils/
│   └── data_downloader.py         # CCXT Daten-Download
├── exchanges/                     # Exchange-Abstraktionsschicht (Phase 2+3 ✅)
│   ├── __init__.py                # Package-Exports (lazy imports für optionale Adapter)
│   ├── models.py                  # InstrumentMeta, OrderResult, PositionInfo, BalanceInfo
│   ├── base.py                    # Abstract ExchangeAdapter Interface
│   ├── binance_adapter.py         # BinanceAdapter — Crypto (ccxt.pro + ccxt sync)
│   ├── oanda_adapter.py           # OandaAdapter — Forex (28 Pairs) + Commodities (4)
│   └── alpaca_adapter.py          # AlpacaAdapter — US Stocks (Top 50)
├── ranker/                        # Cross-Asset Opportunity Ranker (Phase 4 ✅)
│   ├── __init__.py                # Package-Exports
│   ├── universe_scanner.py        # UniverseScanner — scannt ~200 Instrumente
│   ├── opportunity_ranker.py      # OpportunityRanker — Z-Score Ranking
│   └── capital_allocator.py       # CapitalAllocator — Position Limits + Korrelation
└── risk/                          # Circuit Breakers (Phase 5 ✅)
    ├── __init__.py
    └── circuit_breaker.py         # CircuitBreaker — Daily/Weekly/Class Loss Limits
```
