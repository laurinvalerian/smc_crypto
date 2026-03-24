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
| `strategies/smc_multi_style.py` | SMC-Indikatoren (BOS, CHoCH, FVG, OB, Liquidity), Entry-Zone-Erkennung, Multi-Dir Data Loading |
| `rl_brain.py` | Zentrales PPO RL-Gehirn (24-dim Input, shared across alle Instrumente) |
| `filters/` | AAA++ Filter-Module (trend_strength, volume_liquidity, session_filter, zone_quality) |
| `exchanges/` | Exchange-Abstraktionsschicht (Binance, OANDA, Alpaca) |
| `ranker/` | Cross-Asset Opportunity Ranker (Scanner, Ranker, Allocator) |
| `risk/` | Circuit Breakers (Daily/Weekly Loss, Asset-Class Pause, Heat) |
| `config/default_config.yaml` | Alle konfigurierbaren Parameter |
| `utils/forex_data_downloader.py` | OANDA Forex+Commodities OHLCV Download |
| `utils/stock_data_downloader.py` | Alpaca US Stocks OHLCV Download |
| `.env` / `.env.example` | API Keys für alle Broker (nicht committed) |

### Signal-Flow (Top-Down)

```
5m Candle arrives → _prepare_signal()
  ├── Circuit Breaker Check (daily/weekly loss, asset-class pause, heat)
  ├── Volatility Gate (Daily ATR ≥ 0.8%, 5m ATR ≥ 0.15%)
  ├── Volume Pre-Check (≥ 0.5x 20-bar avg)
  ├── Discount/Premium Filter (4H swing range: long only in discount, short only in premium)
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
  ├── Structure-based SL/TP (Liquidity → FVG → OB zones, min RR 2.0)
  ├── Tier-Klassifizierung:
  │   ├── AAA++ (score ≥ 0.88, RR ≥ 3.0, ALLE Komponenten True) → 1-1.5% Risk
  │   └── AAA+  (score ≥ 0.78, RR ≥ 2.0, Kern-Komponenten True) → 0.5-1% Risk
  ├── RL Brain Gate (nach 100 Warmup-Trades)
  └── Bracket Order Execution (Market + SL + TP)
```

### Tier-System (nur AAA++ und AAA+)

**AAA++ (Sniper)** — Score ≥ 0.88, RR ≥ 3.0, erfordert:
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

**AAA+ (Fallback)** — Score ≥ 0.78, RR ≥ 2.0, erfordert:
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
- Tagesverlust ≥ 3% → Stop ALL Trading für 24h (funded: -5% Limit, 2% Buffer)
- Wochenverlust ≥ 5% → `size_reduction_factor = 0.5` (halbe Positionsgrößen)
- Asset-Class Drawdown ≥ 2% → Pause diese Klasse für 12h
- **All-Time Drawdown ≥ 8% → PERMANENTER STOPP** (funded: -10% Limit, 2% Buffer)
- Portfolio Heat > 6% → Keine neuen Positionen
- `can_trade(asset_class)` → `(bool, reason)` Quick-Check vor jedem Entry
- `get_size_factor()` → Multiplikator für Position-Sizing (1.0 oder 0.5)
- Auto-Recovery: Pausen laufen automatisch ab, Size-Reduction hebt sich auf wenn Weekly-PnL erholt
- All-Time DD tracked kumulativ (Peak → Current), erfordert manuellen Reset nach permanentem Stopp

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
- `simulate_trades()` — Komplett umgeschrieben: AAA++ Tier-Gate, Circuit Breaker (inkl. All-Time DD), dynamisches Risk-Sizing (AAA++ 1-2%, AAA+ 0.5-1% von **aktueller Equity** für Zinseszins), Size-Reduction-Faktor von CB, korrekte Position-Sizing (`sl_pct = sl_dist / entry_price`, `position_notional = risk_amount / sl_pct`), Bankrupt-Check bei 10% Equity
- `compute_metrics()` — Erweitert: avg_rr, trades pro Tier, pnl_per_trade, expectancy
- `monte_carlo_check()` — 1000x Trade-Reihenfolge shufflen, 95%-KI berechnen, robust wenn untere Grenze > 0
- `validate_oos_results()` — 4 Gates: PF≥1.5, min 100 Trades, Sharpe≥0.5, Monte Carlo robust
- `check_parameter_stability()` — ±10% Parameter-Perturbation, prüft ob PF-Änderung <50%
- `get_multi_asset_symbols()` — Erkennt automatisch Symbole aus allen 4 Asset-Klassen-Verzeichnissen
- `ASSET_COMMISSION` — Asset-spezifische Kommissionen (Crypto 0.04%, Forex ~0.5 pip, Stocks 0%, Commodities ~1 pip)

**Multi-Asset Backtesting:**
- Lädt Daten aus `data/crypto/`, `data/forex/`, `data/stocks/`, `data/commodities/`
- `symbol_to_asset` Mapping für asset-spezifische CB-Klassen und Kommissionen
- Signals werden per Asset-Klasse gruppiert → korrekte Kommissionen pro Klasse
- Stocks nutzen 5m als Basis-TF (kein 1m verfügbar bei kostenlosen Quellen)
- Circuit Breaker simuliert pro Asset-Klasse separat (2% Klassen-Drawdown → Pause)
- Walk-Forward über alle Asset-Klassen gleichzeitig
- Optuna n_jobs=1 (seriell), Joblib n_jobs=3 (parallel Signals) — verhindert Deadlock auf 4-Core Server
- n_trials=30 pro Window (reduziert von 500 für 4-Core/8GB Server)
- **Signal-Precomputation**: Signale werden EINMAL pro Window mit fixen SMC-Params generiert (alignment_threshold=0), Optuna tuned nur Filter/Trading-Params (alignment_threshold, min_rr, leverage, risk_per_trade)
- **Structure-based TP**: `_find_structure_tp()` findet TP aus Liquidity → FVG → OB Zonen (4H/1H), min RR 2.0, Fallback 3.0R
- **Discount/Premium Filter**: Long nur in Discount (unter 4H Swing-Range Midpoint), Short nur in Premium

**RL Brain im Backtest: NEIN**
- RL Brain wird NICHT im Backtest trainiert (Overfitting-Gefahr)
- Backtest validiert nur die regelbasierten Filter (SMC + AAA++)
- RL Brain trainiert erst im Paper-Trading on-the-fly (nach 100 Warmup-Trades)

**Position Sizing (Compound Growth):**
- Risk ist immer % der **aktuellen Equity** (nicht initial) → Zinseszins-Effekt
- `sl_pct = sl_dist / entry_price` (normalisiert auf %-Basis)
- `position_notional = risk_amount / sl_pct` (korrekte Lots/Shares)
- Hard Cap: max 3% Equity pro Trade
- CB `pnl_pct` trackt gegen **initiales Account** (für funded DD-Limits)
- Bankrupt-Check: Equity < 10% → Trading stoppt

**Trade-Outcome-Modell (V11 — Real Price Path):**
- Walks actual 5m candles forward from entry to determine outcome
- SL/TP hit detection: checks each bar's high/low against current SL and TP
- **Breakeven-Only Stop**: At +1R, SL moves to net-breakeven (entry + 0.1% fee buffer), NO further trailing
- If neither SL nor TP hit within `_MAX_BARS_DEFAULT=576` (48h), trade closes at last bar's close
- Circuit Breaker simuliert Daily/Weekly/All-Time DD Limits pro Trade

**CLI-Flags:** `--monte-carlo`, `--stability-check`

### Daten-Downloader & Broker-Setup (✅ FERTIG)

**Neue Dateien:**
- `utils/forex_data_downloader.py` — OANDA v20 API: 28 Forex-Pairs + 4 Commodities, 1m-1d, Resume-Support, Parquet
- `utils/stock_data_downloader.py` — Alpaca Data API: Top 50 US Stocks, 5m Basis-TF (1m History zu kurz), Parquet
- `.env.example` — Template für alle API Keys (Binance, OANDA, Alpaca)

**Config-Erweiterung (`default_config.yaml`):**
- `exchanges` Section: Binance (testnet), OANDA (practice), Alpaca (paper) mit Env-Var-Referenzen
- `data` Section: Separate Verzeichnisse pro Asset-Klasse (`data/crypto/`, `data/forex/`, `data/stocks/`, `data/commodities/`)

**Daten-Verzeichnisstruktur (✅ ALLE HERUNTERGELADEN):**
```
data/
├── crypto/       # 100+ Coins via CCXT/Binance (1m Basis) — 601 files ✅
├── forex/        # 28 Pairs via OANDA (1m Basis) — 168 files ✅
├── stocks/       # 50 Stocks via Alpaca (5m Basis) — 250 files ✅
└── commodities/  # 4 Instrumente via OANDA (1m Basis) — 24 files ✅
```

**Broker-Accounts:**
- Binance Testnet: ~5000 USDT (nicht änderbar), PnL in % tracken
- OANDA Practice: 100K USD (einstellbar), Live-Preise
- Alpaca Paper: 100K USD (einstellbar), IEX-Feed gratis
- Alle API Keys in `.env` (via `python-dotenv`)

**Balance-Strategie:** Backtesting immer 100K. Paper-Trading: OANDA/Alpaca auf 100K, Binance 5K — Vergleich über %-basierte Metriken (Win Rate, PF, Sharpe, Avg RR).

**Funded Account Ziel (Zukunft):**
- 3× Funded Accounts à 100K: Binance (Crypto), OANDA (Forex+Commodities), Alpaca (Stocks)
- **Max Daily DD: -5%** → Circuit Breaker Limit bei -3% (2% Buffer)
- **Max All-Time DD: -10%** → Circuit Breaker Limit bei -8% (2% Buffer)
- Risk ist **prozentual auf aktuelle Equity** → Zinseszins bei Gewinnen, automatische Reduktion bei Verlusten

### Backtester Bugfixes (2026-03-23)
- **EMA200-Warmup-Bug**: EMA200 braucht 200 Daily Bars, aber Daten starteten erst 2025-03-01 → zu wenige Bars → "neutral" Bias → 0 Trades
  - Fix: `start_date` bleibt `2025-03-01` für Walk-Forward, `history_start: 2024-01-01` für Prefetch
  - `utils/prefetch_history.py` lädt fehlende höhere TFs (1D, 4H, 1H) nach
  - EMA200 bleibt bei 200 (keine Strategie-Kompromisse!)
- **Lookback-Buffer-Bug**: `generate_signals()` schnitt historische Bars ab → höhere TFs ohne Warmup
  - Fix: Lookback-Buffer pro TF (1D: 250, 4H: 100, 1H: 100, 15m: 50), Signals nur innerhalb Window emittiert
- **Timezone-Bug**: Backtester-Windows waren tz-naive, aber Parquet-Daten tz-aware (UTC) → TypeError bei Vergleich → 0 Signale
  - Fix: `generate_signals()` normalisiert start/end automatisch zu tz-aware UTC
- **Alignment-Score-Bug**: Nur 4 von 13 Scoring-Komponenten implementiert → max Score 0.65 → 0 AAA+ Trades
  - Fix: Vollständige 13-Komponenten-Implementierung (bias_strong, h4_confirms, h4_poi, h1_choch, volume_ok)
  - Max Score jetzt 0.90 (bias 0.12 + strong 0.08 + h4 0.08 + h4_poi 0.08 + h1 0.08 + choch 0.06 + zone 0.15 + trigger 0.15 + volume 0.10)
- **Performance-Bug**: 112 Symbole × 30 Trials → 14 min/Trial → 21+ Stunden pro Window
  - Fix: Signal-Precomputation pro Window (einmal generieren, Optuna tuned nur Filter/Trading-Params)
  - Geschätzt: ~15 min Signal-Gen + 30 Trials × ~1s = ~15 min pro Window statt 7+ Stunden
- **Top-30 Crypto**: `get_multi_asset_symbols()` rankt nach 1m-Dateigrösse (Proxy für Liquidität), `max_crypto_symbols: 30`
- **Train-Window**: 3 Monate Train + 1 Monat OOS
- **Test-Ergebnis nach Fixes**: BTC 1718 (73 AAA+, 25 AAA++), EUR_USD 456 (21 AAA+), AAPL 356 (10 AAA+), XAU_USD 153 (2 AAA+)

### Auto Data-Prefetching (`utils/prefetch_history.py`) (✅ FERTIG)
- Prüft alle Instrumente ob genug Higher-TF-Bars VOR `backtest_start` vorhanden (1D: 250, 4H: 500, 1H: 1000)
- Lädt fehlende historische Bars nach (Crypto via CCXT, Forex/Commodities via OANDA, Stocks via Alpaca)
- OANDA-Fix: Benutzt `fromTime` + `count=500` (nicht `fromTime+toTime` — gibt 0 Daten zurück)
- Mergt neue Daten mit bestehenden Parquets (kein Duplikat)
- Sollte VOR Backtest/Paper/Live einmal laufen: `python3 -m utils.prefetch_history`
- Ergebnis: Crypto 300+ Daily Bars, Forex 302+, Stocks 291+, Commodities 301+ (alle vor 2025-03-01) ✅

### Backtester Bugfixes V6 (2026-03-23)
- **Position-Sizing-Bug**: `position_size = risk_amount / sl_dist` normalisierte nicht auf Entry-Preis → Forex SL-Distanzen (0.0050) erzeugten riesige Positionen → -552161% DD
  - Fix: `sl_pct = sl_dist / entry_price`, `position_notional = risk_amount / sl_pct`, PnL als `risk_amount * rr` (Win) bzw. `-risk_amount` (Loss)
- **All-Time DD Breaker**: Neuer permanenter Stopp bei -8% all-time DD (2% Buffer vor funded -10% Limit)
- **Compound Risk**: Risk-Amount basiert auf aktueller Equity (nicht initial) für Zinseszins-Wachstum
- **CB pnl_pct**: Trackt gegen initiales Account-Size (korrekt für funded DD-Limits)

### Backtester V11 Verbesserungen (2026-03-24)

**Grundlegende Änderungen für realistisches SMC-Trading:**

- **Structure-based TP** (`_find_structure_tp()` in `smc_multi_style.py`):
  - TP wird an nächster Liquidity → FVG → OB Zone gesetzt (4H/1H), nicht fixem RR-Multiplier
  - Suchkette: Liquidity Levels → FVG Zonen → Order Blocks → Fallback 3.0 RR
  - Typische Verteilung: ~59% Liquidity, ~28% FVG, ~14% Fallback
  - Min RR 2.0 Gate nach TP-Berechnung

- **Discount/Premium Zone Filter** (`_compute_discount_premium()` in `smc_multi_style.py`):
  - 4H Swing Range → Midpoint berechnen (kein Future-Peek via vlen_4h)
  - Long nur im Discount (unter Midpoint), Short nur im Premium (über Midpoint)
  - Filtert ~30-50% der falschen Signale, fundamentaler SMC/ICT-Grundsatz

- **Breakeven-Only Stop** (kein Trailing mehr):
  - Bei +1R: SL → Net-Breakeven (Entry + 0.1% Fee-Buffer für Fees/Slippage)
  - Danach KEIN weiteres Trailing — Preis hat Raum zum struktur-basierten TP zu laufen
  - Vorher: 1R-Trail → 90% Trailing-Exits, nur 5-10% TP-Hits, avg RR 1.1

- **Max Risk gesenkt**: AAA++ 2.0% → 1.5% (verhindert >10% DD bei schlechtem Timing)

- **Tier-Thresholds gesenkt** (angepasst an Struktur-TPs statt fixem RR):
  - AAA++: min_rr 5.0 → 3.0 (Struktur-TPs geben typisch 2-4 RR)
  - AAA+: min_rr 4.0 → 2.0

- **RR-Parameter**: Optuna tuned `risk_reward` als **min_rr Filter**, nicht als TP-Multiplier

- **Nur Daytrading**: Kein Scalp/Swing, max 48h Holding (576 × 5m Bars)

- **OOS-Validation Gates** (7 Gates):
  1. PF ≥ 1.5
  2. Min 20 Trades (Sniper-Approach: ~6/Monat realistisch)
  3. Sharpe ≥ 0.5
  4. Monte Carlo robust (95%-KI > 0)
  5. Max DD > -10% (funded account safe)
  6. Parameter Stability < 50% PF-Änderung bei ±10%
  7. WR > 20% + positive Expectancy

### Nächste Schritte
1. **Daten**: ✅ FERTIG + Prefetch komplett (Crypto Top 30, Forex 28, Stocks 50, Commodities 4 — alle mit 250+ Daily Bars Warmup)
2. **Backtester V11**: 🔄 Walk-Forward mit Structure-TP + D/P Filter + BE-Only + Risk 1.5%
3. **Paper Trading**: 2 Wochen Demo über alle Asset-Klassen (RL Brain trainiert on-the-fly)
4. **Live → Funded**: 3× Funded Accounts geplant (Binance 100K, OANDA 100K, Alpaca 100K) — max -5% daily DD, max -10% all-time DD

## Testing & Anti-Overfitting

- **Walk-Forward-Validation Pflicht**: 3 Monate Train → 2 Monate Out-of-Sample
- **Out-of-Sample Profit Factor ≥ 1.5**
- **Minimum 20 Trades** im OOS (SMC sniper: ~6-7/Monat)
- **Parameter-Stabilität**: ±10% Änderung darf Performance nicht kippen (Backtester: `check_parameter_stability()`)
- **Monte-Carlo**: R-multiple compound shuffling, 1000x, 95%-KI > 0 (Backtester: `monte_carlo_check()`)
- **7-Gate OOS-Validierung**: PF≥1.5, Trades≥20, Sharpe≥0.5, MC robust, DD>-10%, Stability<50%, Quality (WR>20% + positive expectancy)
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
python3 -m backtest.optuna_backtester --monte-carlo --stability-check

# Daten herunterladen
python3 -m utils.data_downloader --workers 3                  # Crypto (Binance, parallel)
python3 -m utils.forex_data_downloader                        # Forex + Commodities (OANDA)
python3 -m utils.forex_data_downloader --commodities-only     # Nur Commodities
python3 -m utils.stock_data_downloader                        # US Stocks (Alpaca)
python3 -m utils.stock_data_downloader --symbols AAPL MSFT    # Einzelne Stocks

# Status-Monitoring
bash check_downloads.sh                                        # Download-Fortschritt
bash check_backtest.sh                                         # Backtest-Fortschritt
watch -n 30 bash check_backtest.sh                             # Live-Monitor
tail -f backtest/results/backtest.log                          # Detaillierter Log
```

## Wichtige Konstanten

```
ALIGNMENT_THRESHOLD = 0.65       # Minimum Score für Trade-Consideration
TIER_AAA_PLUS_PLUS: score ≥ 0.88, RR ≥ 3.0, risk 1.0-1.5%
TIER_AAA_PLUS:      score ≥ 0.78, RR ≥ 2.0, risk 0.5-1.0%
MIN_DAILY_ATR_PCT = 0.008        # 0.8% Volatility Floor
MIN_5M_ATR_PCT = 0.0015          # 0.15% per 5m Bar
MIN_SL_ATR_MULT = 2.5            # SL mindestens 2.5× ATR
BASE_OBS_DIM = 24                # RL Feature-Dimension
HIDDEN_DIM = 128                 # RL Network Hidden Layer
WARMUP_TRADES = 100              # Trades ohne RL Gate

# Circuit Breaker Limits (funded account safe)
DAILY_LOSS_LIMIT_PCT = 0.03      # -3% daily → full stop 24h (funded: -5%, 2% buffer)
WEEKLY_LOSS_LIMIT_PCT = 0.05     # -5% weekly → halve sizes
ASSET_CLASS_DD_LIMIT = 0.02      # -2% per class → pause 12h
ALLTIME_DD_LIMIT_PCT = 0.08      # -8% all-time → PERMANENT STOP (funded: -10%, 2% buffer)
MAX_PORTFOLIO_HEAT = 0.06        # 6% total open risk
MAX_RISK_PER_TRADE = 0.015       # 1.5% max per trade (AAA++)
```

## Dateistruktur

```
bot/
├── .env                           # API Keys (NICHT in Git!)
├── .env.example                   # Template für .env
├── check_downloads.sh             # Status-Script für Daten-Downloads
├── check_backtest.sh              # Status-Script für Backtest-Fortschritt
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
│   ├── data_downloader.py         # Crypto OHLCV Download (CCXT/Binance, 3 parallel workers)
│   ├── forex_data_downloader.py   # Forex+Commodities OHLCV Download (OANDA v20)
│   └── stock_data_downloader.py   # US Stocks OHLCV Download (Alpaca)
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
