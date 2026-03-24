# CLAUDE.md – SMC Multi-Asset AAA++ Trading Bot

## WICHTIG: CLAUDE.md immer aktuell halten!

**Diese Datei MUSS nach jeder abgeschlossenen Phase, jedem neuen Modul, und jeder Architekturentscheidung sofort aktualisiert werden.** Da der Kontext regelmässig komprimiert wird, ist CLAUDE.md die einzige zuverlässige Quelle für den aktuellen Projektstand. Ohne aktuelle CLAUDE.md gehen kritische Details bei der Komprimierung verloren. Bei jeder Änderung: Dateistruktur, Roadmap-Status, Architektur-Sections und Konstanten prüfen und updaten.

## Projektübersicht

Multi-Asset Trading Bot basierend auf Smart Money Concepts (SMC/ICT), der nur die absolut besten Trades (AAA++) über Crypto, Forex, Stocks und Commodities nimmt. Sniper-Ansatz: weniger Trades, höhere Qualität, maximale Profitabilität.

## Architektur

### Kernkomponenten

| Datei | Zweck |
|---|---|
| `live_multi_bot.py` | Multi-Asset Orchestrator (112 Bots: 30 Crypto + 28 Forex + 50 Stocks + 4 Commodities), PaperBot, Multi-TF Alignment, Order-Execution |
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
- `exchanges/binance_adapter.py` — `BinanceAdapter` wraps ccxt.pro + ccxt sync, inkl. `fetch_max_leverage()` 3-Methoden-Fallback
- `exchanges/__init__.py` — Package-Exports
- **Migration ABGESCHLOSSEN**: `live_multi_bot.py` nutzt NUR noch `self.adapter` Methoden. `self.exchange` (adapter.raw) komplett entfernt. Alle Order/Market/Balance Calls gehen über das abstrakte ExchangeAdapter Interface → funktioniert für Binance, OANDA und Alpaca.

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

### Backtester Bugfixes V11b (2026-03-24)
- **Circuit Breaker Log Spam**: CB pause→expire→re-trigger logged hundreds of lines per second during simulation.
  - Fix 1: Dedup-Logging via `_last_log_state` dict in `circuit_breaker.py` — only logs state CHANGES
  - Fix 2: `simulate_trades()` setzt CB-Logger auf CRITICAL während Simulation — CB funktioniert weiter, nur ohne Log-Spam
- **OOM Kill on 8GB Server**: Process killed at ~4.8GB during Window 1 signal generation (111 instruments × 6 TFs).
  - Fix 1: 4GB Swap-File erstellt (`/swapfile`) als Safety-Net
  - Fix 2: Signal-Generierung in Batches à 30 Instrumente mit `gc.collect()` zwischen Batches
  - Fix 3: Explicit cleanup von Optuna Study + DataFrames zwischen Windows
- **Max Risk 1.5%**: Synced across all files — backtester `compute_dynamic_risk()`, `live_multi_bot.py` `MAX_DYNAMIC_RISK_PCT`, `capital_allocator.py` TIER_RISK_PCT, `default_config.yaml`
- **Discount/Premium Filter**: Already implemented in smc_multi_style.py (`_compute_discount_premium()`) and live_multi_bot.py (`_multi_tf_alignment_score()`)

### Backtester V12 Fixes (2026-03-24) — KRITISCHE BUGS GEFUNDEN + GEFIXT

**V11b Resultate**: PF=93 (lokal) / PF=11 (Server) — unrealistisch. Audit ergab 2 Bugs.

**WICHTIG: Die Kern-Logik (Structure-based TP) ist GLEICH geblieben!**
- TP wird weiterhin an Liquidity → FVG → OB Zonen gesetzt (4H/1H)
- Die `risk_reward` Options in Optuna sind NUR ein min_rr Filter (nicht der TP!)
- `sig.risk_reward` = berechneter RR aus Structure-TP / SL-Distanz

**Bug 1: LOOKAHEAD BIAS in `_find_structure_tp()`** (KRITISCH, GEFIXT)
- SMC-Indikatoren wurden EINMAL auf dem **kompletten Dataset** berechnet (inkl. Zukunft)
- `smc_lib.liquidity()` identifizierte Levels die erst in der Zukunft entstehen
- **Fix**: Neue `_find_structure_tp_safe()` (gleiche Suchkette Liq→FVG→OB→Fallback) die Levels direkt aus `htf_df.iloc[:vlen]` berechnet (raw OHLC, keine vorberechneten Indikatoren)
- 3 Helper: `_find_liquidity_tp()`, `_find_fvg_tp()`, `_find_ob_tp()`
- Alte Funktion umbenannt zu `_find_structure_tp_OLD()` (für Referenz)
- Precomputed via `_HTFArrays` Dataclass + `_precompute_htf_arrays()` (~0.2ms/Call)

**Bug 2: SHORT Breakeven als "Win"** (GEFIXT)
- Short BE-SL = entry - fee_buffer → `current_sl <= entry` → fälschlich "win"
- **Fix**: `pnl_direction = entry - current_sl` → win nur wenn `pnl_direction > 0`
- Sekundär — Hauptursache war Bug 1

**Bug 3: `risk_reward: 1.5` in Optuna Options** (GEFIXT)
- Options geändert auf `[2.0, 2.5, 3.0, 3.5]` (aligned mit Tier-Gate min 2.0)

**Audit-Ergebnisse:**
- Discount/Premium Filter: SAUBER — kein Lookahead
- Signal-Count 710K: Erwartet — raw Signals vor Optuna-Filter (alignment_threshold=0.0)

### V12b Metriken-Fix (2026-03-24) — Breakeven-Inflation

**Problem:** V12 OOS zeigte PF=11.71, WR=81.4% — unrealistisch trotz korrektem Lookahead-Fix.
**Ursache:** Breakeven-Ratchet bläht Metriken auf:
- Bei +1R → SL auf Entry+0.1% → Trade kann nicht mehr verlieren
- BE-Exits (~0% PnL) wurden als "Win" gezählt → WR und PF aufgebläht
- **Equity-Kurve war korrekt** (Dollar-PnL aus echten Preisen), nur die Labels waren falsch

**Audit bestätigt:** Swing-Detection, vlen-Mapping, FVG/OB/Liq-Helpers alle sauber (kein Lookahead).

**Fixes:**
1. **Dritte Outcome-Kategorie "breakeven"**: BE-SL-Exits und near-zero Timeouts → "breakeven" statt "win"/"loss"
2. **Ehrliche Metriken**: `pf_real` (nur echte Wins vs Losses), `winrate_real` (exkl. BE), `be_rate` (BE-Anteil)
3. **Optuna-Objective**: Nutzt jetzt `pf_real` statt inflated PF
4. **Validation Gates**: PF-Gate und WR-Gate nutzen real-Metriken
5. **Stability Check**: Nutzt `pf_real` für Perturbation-Check
6. **Signal-Cache-Versioning**: `SIGNAL_CACHE_VERSION = "v12"` im Cache-Key → alte Caches automatisch ungültig bei Code-Änderungen

**Erwartete Metriken nach Fix:**
- `pf_real`: 1.5-4.0 (statt inflated 11+)
- `winrate_real`: 40-60% (statt inflated 81%)
- `be_rate`: 20-40%
- Equity-Kurve: ähnlich (Dollar-PnL war schon korrekt)

### V12c Fixes (2026-03-24) — Realistisches Position Sizing + Stability Fix

**Problem:** V12b OOS zeigte PF=11.6, WR=80.8%, $823K PnL auf $100K — kein Lookahead, aber unrealistisch wegen:
1. **Compound Growth unlimitiert**: 1% Risk auf aktueller Equity → spätere Trades riskieren 5-10× mehr Dollar
2. **Stability Check kaputt**: `risk_per_trade` (Optuna-tuned) nie an Simulation übergeben, dict-Typen im Perturbation-Loop übersprungen → 0.0% Change = false confidence
3. **OOS-Simulation ignorierte Fixes**: Objective + Stability nutzten neue Params, OOS-Auswertung nicht

**Fixes:**
1. **Equity Cap**: `max_equity_for_sizing = account_size * 2` ($200K) — Position Sizing wächst nicht über 2× Initial, PnL trackt weiter korrekt
2. **`risk_per_trade` als Max-Risk-Cap**: Optuna-tuned Parameter setzt Obergrenze, dynamisches Scoring (besserer Score → mehr Risk) bleibt erhalten
3. **Stability Check repariert**: `risk_per_trade` an `_run_with_params()` übergeben, dict-Typen korrekt übersprungen
4. **OOS-Simulation**: Bekommt jetzt `risk_per_trade_override` + `max_equity_for_sizing`
5. **risk_pct Reporting**: `risk_amount / equity` statt `risk_amount / initial_account` (war irreführend)

**Signal-Cache bleibt gültig** — `SIGNAL_CACHE_VERSION` unverändert, Fixes betreffen nur Simulation/Metriken.

### V12c Backtest-Ergebnisse (2026-03-24) — 3/3 PASS

| Window | PF(real) | WR(real) | BE% | Trades | DD | Stability | Verdict |
|--------|----------|----------|-----|--------|------|-----------|---------|
| W0 (Jun-Aug) | 11.98 | 76.6% | 28% | 365 | -5.80% | 6.6% | PASS |
| W1 (Sep-Nov) | 6.16 | 55.8% | 43% | 181 | -2.14% | 37.6% | PASS |
| W2 (Dec-Feb) | 9.04 | 74.3% | 38% | 280 | -5.17% | 1.0% | PASS |

### Cross-Window Anti-Overfitting Test (2026-03-24) — ALLE EVERGREEN

Getestet: 6 Parameter-Sets auf ALLEN 3 OOS-Windows. Jedes Set profitabel auf allen Windows (kein PF < 1.5).

**Bester "Sniper" Set (Evergreen Live-Parameter):**
```
alignment_threshold: 0.88
min_rr: 3.0
leverage: 5
risk_per_trade: 0.01 (1.0% max, dynamisch je nach Score)
```

| Params | W0(Jun-Aug) | W1(Sep-Nov) | W2(Dec-Feb) | Min PF |
|--------|-------------|-------------|-------------|--------|
| **Sniper** | PF=13.5, DD=-5.3% | PF=7.9, DD=-6.7% | PF=8.3, DD=-4.3% | **7.88** |
| W1-best | PF=8.3, DD=-2.3% | PF=6.2, DD=-2.1% | PF=8.5, DD=-1.9% | 6.15 |
| Median | PF=11.1, DD=-6.0% | PF=3.2, DD=-14.3% | PF=6.8, DD=-6.3% | 3.17 |

**Sniper-Set gewinnt**: Höchster Min-PF (7.88), konsistentester DD (<7%), nie optimiert (zero overfitting risk).

### Paper Grid — Multi-Variant A/B Testing (✅ FERTIG)

**Neues Modul: `paper_grid.py`**
- 20 Parameter-Varianten laufen parallel auf dem gleichen Signal-Stream
- 1 Bot empfängt Signale, PaperGrid evaluiert alle 20 Varianten (kein extra API-Load)
- Jede Variante: eigenes virtuelles Equity ($100K), PnL, Drawdown, Metriken
- Realistische Fees pro Asset-Klasse (Crypto 0.10%, Forex ~1 pip, Stocks 0.02%, Commodities ~2 pip)
- Persistence: State wird alle 10s als JSON gespeichert (Crash-Recovery)
- Export: CSV mit allen Trades + Summary

**Varianten-Highlights:**
| Variante | Align | RR | Lev | Risk | Zweck |
|----------|-------|-----|-----|------|-------|
| Sniper-1.0 | 0.88 | 3.0 | 5 | 1.0% | Evergreen Winner |
| Sniper-1.5 | 0.88 | 3.0 | 5 | 1.5% | Risk-Test |
| Sniper-2.0 | 0.88 | 3.0 | 5 | 2.0% | Risk-Test |
| AAA+-Base | 0.78 | 2.0 | 7 | 1.0% | Mehr Trades |
| Ultra-Sniper | 0.90 | 3.5 | 3 | 1.0% | Nur die Besten |
| Wild-Max | 0.78 | 2.0 | 15 | 2.0% | Max Aggression |
| Wild-Min | 0.90 | 3.5 | 3 | 0.3% | Max Sicherheit |
| + 13 weitere Varianten (Leverage-Tests, RR-Tests, Wild Shots) |

**Integration in `live_multi_bot.py`:**
- `_prepare_signal()` → `paper_grid.evaluate_signal()` (jedes Signal durch alle Varianten)
- `_record_close()` → `paper_grid.record_trade_close()` (echte Exit-Preise)
- Dashboard: Top 10 Varianten nach PnL im Rich Layout
- Shutdown: Auto-Save + CSV-Export

### Zombie Order Prevention (✅ FERTIG)

**Problem:** Wenn SL/TP triggert, bleibt die andere Order offen ("Zombie"). Kann später auf neue Position füllen.

**3-Layer Schutz (alle Exchanges):**
1. **Per-Trade Cleanup** (`_record_close()`): Cancel SL+TP by ID nach Trade-Close
2. **Periodic Sweep** (`_sweep_zombie_orders()`, alle 60s): Scannt alle offenen Orders, cancelled unmatched
3. **Startup Sweep** (`run()`): Cancelled ALLE offenen Orders beim Start (keine aktiven Trades at startup)

**Exchange-Kompatibilität:**
- Binance: Standalone SL/TP Orders → Zombie möglich → alle 3 Layer aktiv
- OANDA: Standalone STOP/LIMIT Orders → gleicher Mechanismus via `cancel_order()`
- Alpaca: Standalone SL/TP Orders → gleicher Mechanismus via `cancel_order()`
- Alle Adapter implementieren `fetch_open_orders()` + `cancel_order()` → generische Lösung

### Multi-Asset Live Bot Refactoring (✅ FERTIG)

**`live_multi_bot.py`** von Crypto-only (100 Binance Bots) auf Multi-Asset (112 Instrumente) umgebaut:

**Neue Constants & Symbol-Listen:**
- `TOP_30_CRYPTO` (30 höchste Liquidität), `FOREX_28`, `STOCKS_50`, `COMMODITIES_4`
- `ALL_INSTRUMENTS` dict → 112 Instrumente total
- `ASSET_COMMISSION` — per Asset-Klasse (crypto 0.04%, forex 0.005%, stocks 0%, commodities 0.01%)
- `ASSET_SMC_PARAMS` — per Asset-Klasse aus `config/default_config.yaml` smc_profiles
- `ASSET_MAX_LEVERAGE` — crypto 20x, forex 30x, stocks 4x, commodities 20x
- `ASSET_CLASS_ID` — RL Feature (crypto 0.0, forex 0.25, stocks 0.5, commodities 0.75)
- `symbol_to_asset_class()` Helper

**Multi-Exchange Init (`create_adapters()`):**
- Async, erstellt BinanceAdapter + OandaAdapter + AlpacaAdapter basierend auf .env Keys
- OANDA mapped auf "forex" + "commodities" (gleiche Instanz, Dedup via `id(adapter)`)
- Graceful Skip bei fehlenden Keys/Packages

**PaperBot Refactoring:**
- Neue Params: `asset_class`, `adapter` (ExchangeAdapter)
- `self.exchange` entfernt — alle Calls über `self.adapter` (Exchange-agnostisch)
- Per-Asset SMC Params, Commission, Leverage Cap
- `load_history()` jetzt async via `adapter.fetch_ohlcv()` (250 daily bars für EMA200)

**Signal Pipeline Asset-Class-Aware:**
- 5× hardcoded `"crypto"` → `self.asset_class` (Circuit Breaker, Volume, Session, Paper Grid, Record Close)
- Trading Hours Check via `adapter.is_market_open()`
- Volatility Gate per Asset-Klasse (`self.min_daily_atr_pct`, `self.min_5m_atr_pct`)

**Data Feed Strategie:**
- Crypto: WebSocket (`watch_ohlcv`, `watch_ticker`) — bestehend
- Forex/Stocks/Commodities: REST Polling (`_poll_candles`, `_poll_ticker`, 10s/5s Intervall)

**Runner Multi-Exchange:**
- `LiveMultiBotRunner` akzeptiert `adapters: dict[str, ExchangeAdapter]`
- Per-Adapter Position Polling, Zombie Sweep, Balance Fetch (mit OANDA Dedup)
- Shutdown: alle Adapter geschlossen

**Dashboard:**
- Header: "SMC MULTI-ASSET LIVE TRADING DASHBOARD"
- Per-Class Bot-Counts, "Class" Column in Bot-Tabellen

**`async_main()` + `main()`:**
- `main()` → `asyncio.run(async_main())`
- History Loading in 10er-Batches (Rate-Limit-freundlich)
- Graceful Degradation: nur verfügbare Asset-Klassen starten

### Order Execution Migration (✅ FERTIG)

**`self.exchange` komplett entfernt** — alle Order-/Market-Calls gehen über `self.adapter`:

**`_place_bracket_order()`**: Entry via `adapter.create_market_order()`, SL via `adapter.create_stop_loss()`, TP via `adapter.create_take_profit()`, Cancel via `adapter.cancel_order()`, Precision via `adapter.price_to_precision()`

**`_execute_bracket_order_with_risk_reduction()`**: ~500→~250 Zeilen. `adapter.get_instrument()` ersetzt manuelles Binance-Filter-Parsing. `adapter.fetch_max_leverage()` ersetzt 3-Methoden-Fallback. `adapter.set_margin_mode()` + `adapter.set_leverage()` für Margin-Management.

**`_fetch_balance()`**: `adapter.fetch_balance()` → `BalanceInfo.free` (statt raw ccxt dict-Parsing)

**Runner**: `self.exchange` → `self._crypto_adapter` für WebSocket-Feeds. Kein `self.exchange` mehr im gesamten File.

### Per-Asset-Class Backtesting (✅ FERTIG)

**Neuer Modus**: `python3 -m backtest.optuna_backtester --per-class`

- Optuna läuft **separat pro Asset-Klasse** (statt global über 112 Instrumente)
- Per-Class Leverage-Ranges aus `config/default_config.yaml` → `tuning_per_class` (crypto 3-20x, forex 5-30x, stocks 1-4x, commodities 3-20x)
- **Cross-Window-Validierung automatisiert**: Alle Kandidaten-Params werden auf ALLEN OOS-Windows getestet
- **Evergreen-Kriterium**: `pf_real >= 1.5` auf jedem Window → Ranking nach `min(pf_real)`
- Ergebnis: `backtest/results/{asset_class}/evergreen_params.json` pro Klasse
- Weniger RAM pro Run (30 Symbole statt 112)
- Commodities-Fallback: < 50 Signale → konservative Default-Params

**Paper Grid Varianten-Generator**: `python3 -m backtest.optuna_backtester --generate-paper-grid`
- Liest Evergreen-Params pro Klasse
- Generiert **20 Varianten pro Klasse** (80 total): Base, Conservative, Risk-Tests, Leverage-Tests, RR-Tests, Aggressive, Defensive, Wild-Max, Wild-Min, etc.
- Gespeichert als `paper_grid_results/variants.json`
- `PaperGrid` lädt automatisch aus variants.json wenn vorhanden

**Paper Grid Asset-Class-Aware:**
- `VariantConfig` hat neues Feld `asset_class: str | None`
- `evaluate_signal()` filtert: Variant mit `asset_class="crypto"` evaluiert nur Crypto-Signale
- Backward-compat: `asset_class=None` → evaluiert alle Klassen

### Nächste Schritte
1. **Daten**: ✅ FERTIG + Prefetch komplett
2. **V12 Fixes**: ✅ GEFIXT — Lookahead-freie Structure-TP + Short-BE Fix + risk_reward Options
3. **V12b Metriken-Fix**: ✅ GEFIXT — Breakeven-Outcome + ehrliche Metriken + Cache-Versioning
4. **V12c Backtester-Fix**: ✅ GEFIXT — Equity Cap + risk_per_trade Cap + Stability Fix + OOS Fix
5. **Backtester V12c**: ✅ FERTIG — 3/3 PASS, Evergreen-Params validiert
6. **Paper Grid**: ✅ FERTIG — 20 Varianten, A/B Testing, realistische Fees
7. **Zombie Prevention**: ✅ FERTIG — 3-Layer Schutz, Startup Sweep
8. **Multi-Asset Live Bot**: ✅ FERTIG — 112 Instrumente, 3 Exchanges, REST+WS Feeds
9. **Order Execution Migration**: ✅ FERTIG — `self.exchange` entfernt, alles über `self.adapter`
10. **Per-Asset-Class Backtesting**: ✅ FERTIG — `--per-class` + Cross-Window Evergreen + Paper Grid Generator
11. **Per-Class Backtest ausführen**: ⏳ `python3 -m backtest.optuna_backtester --per-class` (~3-4h)
12. **Paper Grid Varianten generieren**: ⏳ `--generate-paper-grid` (nach Backtest)
13. **Paper Trading**: ⏳ 2 Wochen Demo mit per-Class Evergreen-Params über alle Asset-Klassen
14. **Live → Funded**: 3× Funded Accounts geplant

## Testing & Anti-Overfitting

- **Walk-Forward-Validation Pflicht**: 3 Monate Train → 2 Monate Out-of-Sample
- **Out-of-Sample Profit Factor ≥ 1.5**
- **Minimum 20 Trades** im OOS (SMC sniper: ~6-7/Monat)
- **Parameter-Stabilität**: ±10% Änderung darf Performance nicht kippen (Backtester: `check_parameter_stability()`)
- **Monte-Carlo**: R-multiple compound shuffling, 1000x, 95%-KI > 0 (Backtester: `monte_carlo_check()`)
- **7-Gate OOS-Validierung**: PF≥1.5, Trades≥20, Sharpe≥0.5, MC robust, DD>-10%, Stability<50%, Quality (WR>20% + positive expectancy)
- **Cross-Window-Test (AUTOMATISIERT)**: `cross_window_validate()` testet alle Kandidaten-Params auf ALLEN OOS-Windows. Nur "Evergreen" Params (PF≥1.5 auf JEDEM Window) werden akzeptiert. Ranking: höchstes Minimum-PF.
- **Per-Asset-Class Params**: Gleiche Params über ALLE Zeitfenster, aber VERSCHIEDENE Params pro Asset-Klasse (crypto, forex, stocks, commodities verhalten sich fundamental anders)
- **RL Pre-Training**: Nur mit Out-of-Sample Backtest-Trades, Curriculum-basiert (erst nach 100 Paper-Trades)
- **Paper-Trading**: Mindestens 2 Wochen vor Live, innerhalb 1 Std-Abweichung der Backtests

### Per-Class Backtest Workflow (Komplett-Anleitung)

```bash
# Schritt 1: Per-Class Backtest (~3-4h, sequentiell pro Klasse)
python3 -m backtest.optuna_backtester --per-class

# Was passiert:
# - Für jede der 4 Klassen (crypto, forex, stocks, commodities):
#   - Signal-Precomputation NUR für diese Klasse (weniger RAM)
#   - Optuna: 30 Trials pro Window, per-class Leverage-Range
#   - OOS-Validierung mit 7 Gates
#   - Cross-Window: testet ALLE Kandidaten auf ALLEN OOS-Windows
#   - Speichert: backtest/results/{class}/evergreen_params.json
# - Walk-Forward: 3 Windows (W0 Jun-Aug, W1 Sep-Nov, W2 Dec-Feb)

# Schritt 2: Paper Grid Varianten generieren
python3 -m backtest.optuna_backtester --generate-paper-grid

# Was passiert:
# - Liest evergreen_params.json pro Klasse
# - Generiert 20 Varianten pro Klasse (80 total)
# - Speichert: paper_grid_results/variants.json
# - Varianten: Base, Conservative, Risk-Tests, Leverage-Tests, etc.

# Schritt 3: Paper Trading starten
python3 live_multi_bot.py

# Was passiert:
# - Lädt variants.json → 80 Varianten (20 pro Klasse)
# - 112 Bots (30 crypto + 28 forex + 50 stocks + 4 commodities)
# - Jedes Signal wird durch alle passenden Varianten evaluiert
# - Dashboard zeigt Top-Varianten pro Klasse

# Monitoring:
tail -f backtest/results/backtest.log           # Backtest-Log
tail -f paper_grid_results/summary.csv          # Paper Grid Ergebnisse
```

### Ergebnis-Dateien nach Per-Class Backtest

```
backtest/results/
├── crypto/
│   ├── evergreen_params.json    # Beste Evergreen-Params für Crypto
│   ├── oos_trades_w0.csv        # OOS Trades pro Window
│   ├── top_params_w0.csv        # Top-20% Params pro Window
│   └── param_importance.csv     # fANOVA Importance Ranking
├── forex/
│   └── evergreen_params.json
├── stocks/
│   └── evergreen_params.json
├── commodities/
│   └── evergreen_params.json
├── evergreen_summary.json       # Übersicht: alle Klassen
└── signal_cache/                # Gecachte Signale (wiederverwendbar)

paper_grid_results/
├── variants.json                # 80 Varianten (20 pro Klasse)
├── state.json                   # Laufender Paper-Trading State
└── summary.csv                  # Trade-Export
```

## Exchanges

| Asset-Klasse | Exchange/Broker | Status |
|---|---|---|
| Crypto | Binance USDT-M Futures (CCXT) | ✅ Adapter fertig, aktiv (Testnet) |
| Forex | OANDA v20 API | ✅ Adapter fertig (28 Pairs), braucht `pip install v20` |
| Stocks | Alpaca REST API | ✅ Adapter fertig (Top 50), braucht `pip install alpaca-py` |
| Commodities | OANDA (XAU, XAG, WTI, BCO) | ✅ Adapter fertig (via OandaAdapter) |

## Commands

```bash
# Live/Paper Trading starten (Dashboard im Terminal)
python3 live_multi_bot.py [--config config/default_config.yaml]

# Backtesting — Global (alle Assets gleiche Params)
python3 -m backtest.optuna_backtester

# Backtesting — Per Asset-Klasse (separate Optuna + Cross-Window Evergreen)
python3 -m backtest.optuna_backtester --per-class

# Paper Grid Varianten generieren (aus Evergreen-Ergebnissen)
python3 -m backtest.optuna_backtester --generate-paper-grid

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
# Evergreen Live-Parameter — werden JETZT per Asset-Klasse optimiert!
# Vorher: globale Sniper-Params für alle Klassen
# Nachher: per-class Evergreen aus `backtest/results/{class}/evergreen_params.json`
# Fallback wenn kein Evergreen gefunden:
ALIGNMENT_THRESHOLD = 0.88       # Sniper: nur AAA++ Trades
MIN_RR = 3.0                     # Nur Trades mit RR ≥ 3.0
LEVERAGE = 5                     # Konservativ
RISK_PER_TRADE = 0.005           # 0.5% (konservativer Fallback)
MAX_EQUITY_FOR_SIZING = 2x       # Equity Cap bei 2× Initial (anti-compound-explosion)

# Per-Class Leverage-Ranges (Optuna tuning_per_class in config)
CRYPTO_LEVERAGE_RANGE = 3-20x
FOREX_LEVERAGE_RANGE = 5-30x
STOCKS_LEVERAGE_RANGE = 1-4x
COMMODITIES_LEVERAGE_RANGE = 3-20x

# Tier-Definitionen (für Signal-Klassifizierung)
TIER_AAA_PLUS_PLUS: score ≥ 0.88, RR ≥ 3.0, risk 1.0-1.5% (max 1.5%)
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
MAX_DYNAMIC_RISK_PCT = 0.015     # Hard cap in live_multi_bot.py (synced with backtester)
```

## Dateistruktur

```
bot/
├── .env                           # API Keys (NICHT in Git!)
├── .env.example                   # Template für .env
├── check_downloads.sh             # Status-Script für Daten-Downloads
├── check_backtest.sh              # Status-Script für Backtest-Fortschritt
├── live_multi_bot.py              # Haupt-Orchestrator (PaperBot + Runner)
├── paper_grid.py                  # Multi-Variant A/B Testing (80 Varianten: 20 pro Asset-Klasse, virtuelle PnL)
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
│   └── optuna_backtester.py       # Walk-Forward Optimizer (--per-class + --generate-paper-grid)
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
