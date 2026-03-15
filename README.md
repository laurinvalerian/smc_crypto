# SMC Crypto Trading Bot

Automated **Smart Money Concepts (SMC / ICT)** trading bot for Binance USDT-M Futures.  
**Final architecture:** exactly **100 bots**, each permanently specialised on one coin from the Top 100 Evergreen list, with identical fixed SMC parameters and per-bot PPO RL brain.

---

## Features

| Feature | Description |
|---|---|
| **100 Coin-Specialised Bots** | 1 bot = 1 coin (fixed 1:1 mapping from Top 100 Evergreen list) |
| **Fixed SMC Parameters** | swing_length=10, fvg_threshold=0.00045, order_block_lookback=28, liquidity_range=0.75%, alignment=0.52 |
| **Fixed Money Management** | Risk 1%, R:R min 1:2, full SL or full TP (no partials) |
| **Per-Bot RL Brain (PPO)** | Each bot has its own PPO agent that learns a yes/no trade filter. Reward = pure PnL% |
| **SMC Indicators** | FVG, Order Blocks, BOS/CHoCH, Liquidity Pools, Swing Highs/Lows |
| **Rich Live Dashboard** | Top 20 / Worst 20 bots, WebSocket status, total equity & PnL |
| **WebSocket Auto-Reconnect** | Up to 5 retries with exponential backoff per symbol |

---

## Project Structure

```
smc_crypto/
├── config/
│   └── default_config.yaml      # Base configuration
├── data/                         # Downloaded OHLCV Parquet files
├── strategies/
│   └── smc_multi_style.py       # Core SMC multi-style strategy
├── backtest/
│   ├── optuna_backtester.py     # Walk-forward Optuna optimiser
│   └── results/                 # CSV, JSON, plots
├── utils/
│   └── data_downloader.py       # CCXT downloader + volume ranking
├── live_multi_bot.py            # 100-bot live system (main entry point)
├── rl_brain.py                  # Per-bot PPO RL brain
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start – 100-Bot Live System

### 1. Prerequisites

- **Python 3.11+** (tested with 3.11 and 3.12)
- **pip** or **uv** package manager
- Binance Futures Testnet API keys

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

### 3. Set up API keys

```bash
cp .env.example .env
# Edit .env and fill in your Binance Testnet keys:
#   BINANCE_API_KEY=your_testnet_api_key
#   BINANCE_SECRET=your_testnet_secret
```

### 4. Launch the 100-bot system

```bash
python live_multi_bot.py
```

That's it! The system will:
1. Create **99 coin-specialised bots** (one per coin from the Top 100 Evergreen list)
2. Connect via WebSocket to Binance Futures Testnet
3. Each bot monitors 5m candles for its assigned coin only
4. Entry decisions: SMC alignment score ≥ 0.52 **AND** RL brain says "trade"
5. Display a Rich live dashboard with Top 20 / Worst 20 bots + WebSocket status
6. `Ctrl+C` → graceful shutdown with final summary

### Optional CLI arguments

```bash
python live_multi_bot.py --config config/default_config.yaml --output-dir live_results
```

### 5. Output files

```
live_results/
├── bot_001_equity.csv    # Per-bot equity curve
├── bot_001.log           # Per-bot trade log
├── ...
├── bot_099_equity.csv
├── bot_099.log
└── rl_models/            # Per-bot PPO model checkpoints
    ├── bot_001_ppo.pt
    ├── ...
    └── bot_099_ppo.pt
```

---

## Architecture

### Fixed Parameters (all 100 bots identical)

| Parameter | Value |
|---|---|
| swing_length | 10 |
| fvg_threshold | 0.00045 |
| order_block_lookback | 28 |
| liquidity_range_percent | 0.0075 |
| alignment_threshold | 0.52 |
| weight_day | 1.25 |
| bos_choch_filter | medium |
| Risk per trade | 1% |
| Risk-Reward | min 1:2 |
| ATR Period | 14 |
| EMA | 20 / 50 |
| Min Volume Filter | 1.0× average |

### RL Brain (PPO)

Each bot has its own lightweight PPO (Proximal Policy Optimization) agent:
- **Observation**: 12-dim vector (alignment score, ATR, EMAs, volume, returns, RSI)
- **Action**: Binary – 0 = skip, 1 = take the trade
- **Reward**: Pure PnL change in % (no shaping, no R:R bonus)
- **Update**: Mini-batch PPO with clipped surrogate objective every 256 decisions
- **Persistence**: Model saved/loaded per bot (`rl_models/bot_NNN_ppo.pt`)

The brain acts as a gating filter on top of the rule-based SMC strategy.  
If PyTorch is not installed, the brain falls back to pass-through mode (all trades taken).

---

## Backtesting (Optional)

### Download historical data

```bash
python -m utils.data_downloader
```

### Run the backtest

```bash
python -m backtest.optuna_backtester
```

### Optuna Dashboard

```bash
optuna-dashboard sqlite:///backtest/results/optuna_study.db
```

---

## Environment Variables

Create a `.env` file in the project root (or copy `.env.example`):

```env
# Binance Futures Testnet
BINANCE_API_KEY=your_testnet_api_key
BINANCE_SECRET=your_testnet_secret
```

---

## License

Private – not for redistribution.