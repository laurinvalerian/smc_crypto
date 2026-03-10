# SMC Crypto Trading Bot

Automated **Smart Money Concepts (SMC / ICT)** trading bot for Binance USDT-M Futures with multi-style execution (Scalp / Day / Swing), walk-forward Optuna optimisation, and production-ready architecture.

---

## Features

| Feature | Description |
|---|---|
| **Multi-Style** | Automatically selects Scalp (1 m/5 m), Day (15 m/1 h), or Swing (4 h/1 D) based on top-down alignment score |
| **SMC Indicators** | FVG, Order Blocks, BOS/CHoCH, Liquidity Pools, Swing Highs/Lows via [`smartmoneyconcepts`](https://github.com/joshyattridge/smart-money-concepts) |
| **Walk-Forward Optimisation** | 6-month train / 3-month test rolling windows with Optuna Bayesian search (≥ 2 000 trials) |
| **Dynamic Volume Filter** | Historical 30-day rolling volume ranking – only trade top-100 coins |
| **Risk Management** | Exact position sizing (risk % × leverage × SL distance), hard leverage cap at 15× |
| **Full Metrics** | Profit Factor, Max Drawdown, Sharpe, Winrate, Recovery Factor, per-style PnL |

---

## Project Structure

```
smc_crypto/
├── config/
│   └── default_config.yaml      # All tunable parameters
├── data/                         # Downloaded OHLCV Parquet files
├── strategies/
│   └── smc_multi_style.py       # Core SMC multi-style strategy
├── backtest/
│   ├── optuna_backtester.py     # Walk-forward Optuna optimiser
│   └── results/                 # CSV, JSON, plots
├── utils/
│   └── data_downloader.py       # CCXT downloader + volume ranking
├── live/                         # Phase 2 – live/paper trading
├── docker/                       # Phase 2 – containerisation
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start (MacBook M4/M5 or Hetzner VPS)

### 1. Prerequisites

- **Python 3.11+** (tested with 3.11 and 3.12)
- **pip** or **uv** package manager
- (Optional) Binance Futures Testnet API keys for paper trading

### 2. Clone & install

```bash
git clone https://github.com/laurinvalerian/smc_crypto.git
cd smc_crypto

# Create virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Edit `config/default_config.yaml` to adjust parameters. Defaults are:

| Parameter | Default | Range |
|---|---|---|
| Account size | 100 000 USDT | Fixed |
| Leverage | 5–15× | Optuna-tuned |
| Risk per trade | 0.3–1.5 % | Optuna-tuned |
| Risk-reward | 1:2 – 1:5 | Optuna-tuned |
| Volume filter | Top 100 | Configurable |

### 4. Download historical data

```bash
python -m utils.data_downloader
```

This downloads all available Binance Futures 1 m OHLCV data since November 2019 (~6.5 years) for every USDT-perp symbol, resamples to 5 m / 15 m / 1 h / 4 h / 1 D, computes the historical 30-day volume ranking, and saves everything as compressed Parquet in `data/`.

> **Note:** Full download may take several hours. The downloader supports resumption – re-run the command to continue where it left off.

### 5. Run the backtest

```bash
python -m backtest.optuna_backtester
```

The backtester will:

1. Generate rolling walk-forward windows (6-month train / 3-month test).
2. Run Optuna Bayesian optimisation with ≥ 2 000 trials per window.
3. Test the best parameters out-of-sample.
4. Extract the **top 20 %** best parameter sets per window.
5. Compute **parameter importance** (fANOVA) and save plots + CSV.
6. Save all results to `backtest/results/`.

### 6. Review results

```
backtest/results/
├── wfo_summary.csv           # Per-window OOS metrics
├── global_top_params.csv     # Best parameter sets across all windows
├── top_params_w0.csv         # Top 20% for window 0
├── top_params_w1.csv         # Top 20% for window 1
├── ...
├── param_importance.csv      # Parameter importance ranking
├── param_importance.html     # Interactive importance chart
└── backtest_stats.json       # Aggregate summary statistics
```

### 7. (Optional) Optuna Dashboard

```bash
optuna-dashboard sqlite:///backtest/results/optuna_study.db
```

Opens a web UI at `http://localhost:8080` where you can explore trials, parameter distributions, and intermediate results.

---

## Environment Variables

Create a `.env` file in the project root (or copy `.env.example`):

```env
# Binance Futures Testnet (Phase 2 – paper trading)
BINANCE_API_KEY=your_testnet_api_key
BINANCE_SECRET=your_testnet_secret

# Optional: Binance Futures Live (Phase 3)
# BINANCE_LIVE_API_KEY=...
# BINANCE_LIVE_SECRET=...
```

---

## Roadmap

| Phase | Status | Description |
|---|---|---|
| **1. Extensive Backtesting** | 🔄 Current | Walk-forward Optuna optimisation across 6.5 years of data |
| **2. Paper Trading** | ⏳ Next | Top 20 % configs on Binance Testnet (100 parallel) |
| **3. Live Selection** | ⏳ Planned | Pick winners that excel in both backtest and paper trading |
| **4. RL Brain** | ⏳ Planned | Small RL model as yes/no signal filter on the best strategy |

---

## Key Design Decisions

- **100 000 USDT fixed account** for standardised comparison across parameter sets.
- **Leverage hard-capped at 15×** to avoid funding-rate drag and liquidation risk.
- **30-day volume ranking computed historically** so backtest results are not biased by survivorship.
- **No RL in Phase 1** – the bot uses pure rule-based SMC logic first; RL is added later as a lightweight signal filter.

---

## License

Private – not for redistribution.