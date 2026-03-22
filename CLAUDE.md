# CLAUDE.md – SMC Multi-Asset AAA++ Trading Bot

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
| `exchanges/` | Exchange-Abstraktionsschicht (ExchangeAdapter Interface + BinanceAdapter) |
| `config/default_config.yaml` | Alle konfigurierbaren Parameter |

### Signal-Flow (Top-Down)

```
5m Candle arrives → _prepare_signal()
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

### Phase 4: Cross-Asset Opportunity Ranker (NÄCHSTE)
- `ranker/opportunity_ranker.py` — Z-Score-normalisiertes Ranking über alle Assets
- `ranker/capital_allocator.py` — Max 5 Positionen, Korrelations-Check (>0.7 = nur den besseren)
- `ranker/universe_scanner.py` — ~200 Instrumente dynamisch scannen
- 100 fixe Bots → Pool von InstrumentScanner-Objekten

### Phase 5: Circuit Breakers
- Tagesverlust -3% → Stop 24h
- Wochenverlust -5% → Halbe Positionsgrößen
- Asset-Class Drawdown -2% → Pause 12h
- Max Portfolio Heat 6%

## Testing & Anti-Overfitting

- **Walk-Forward-Validation Pflicht**: 3 Monate Train → 1 Monat Out-of-Sample
- **Out-of-Sample Profit Factor ≥ 1.5**
- **Minimum 100 Trades** im OOS für statistische Relevanz
- **Parameter-Stabilität**: ±10% Änderung darf Performance nicht kippen
- **Monte-Carlo**: Trade-Reihenfolge 1000x shufflen, 95%-KI muss profitabel sein
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
└── ranker/                        # (Phase 4 — noch nicht erstellt)
```
