"""
Paper Grid — Multi-Variant A/B Testing for Paper Trading.

Runs 20 parameter variants in parallel on the same live signal stream.
Each variant filters signals independently and tracks its own virtual PnL.
No extra API calls — signals come once, get evaluated 20 times.

Usage:
    grid = PaperGrid()  # loads default 20 variants
    # On each signal:
    decisions = grid.evaluate_signal(signal_dict, asset_class="crypto")
    # On trade close:
    grid.record_trade_close(variant_name, trade_id, exit_price)
    # Dashboard:
    rows = grid.dashboard_data()
"""
from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("paper_grid")

# ═══════════════════════════════════════════════════════════════════
#  Asset-class fees (same as backtester ASSET_COMMISSION)
# ═══════════════════════════════════════════════════════════════════

# Phase 2.1 SSOT (2026-04-18): values imported from core.constants.
# Crypto-only after Phase 1 strip — dict form retained for caller compatibility.
from core.constants import COMMISSION, SLIPPAGE
ASSET_COMMISSION: dict[str, float] = {"crypto": COMMISSION}
SLIPPAGE_PCT = SLIPPAGE


# ═══════════════════════════════════════════════════════════════════
#  Variant configuration + state
# ═══════════════════════════════════════════════════════════════════

@dataclass
class VariantConfig:
    name: str
    alignment_threshold: float
    min_rr: float
    leverage: int
    risk_per_trade: float  # max risk cap (dynamic scoring still applies)
    asset_class: str | None = None  # None = evaluate for all classes


@dataclass
class VirtualTrade:
    trade_id: str
    symbol: str
    direction: str          # "long" | "short"
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    position_notional: float
    risk_amount: float
    cost: float             # total fees (entry + exit)
    asset_class: str
    entry_time: float       # time.time()
    score: float


@dataclass
class VariantState:
    config: VariantConfig
    equity: float = 100_000.0
    peak_equity: float = 100_000.0
    trades: list[dict] = field(default_factory=list)     # completed trades
    open_trades: dict[str, VirtualTrade] = field(default_factory=dict)  # trade_id → VirtualTrade
    _trade_counter: int = 0

    # ── Metrics (live-updated) ──────────────────────────────────
    total_pnl: float = 0.0
    n_wins: int = 0
    n_losses: int = 0
    n_breakeven: int = 0

    @property
    def n_trades(self) -> int:
        return self.n_wins + self.n_losses + self.n_breakeven

    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.equity - self.peak_equity) / self.peak_equity

    @property
    def winrate_real(self) -> float:
        real = self.n_wins + self.n_losses
        return self.n_wins / real if real > 0 else 0.0

    @property
    def pf_real(self) -> float:
        win_pnl = sum(t["pnl"] for t in self.trades if t["outcome"] == "win")
        loss_pnl = abs(sum(t["pnl"] for t in self.trades if t["outcome"] == "loss"))
        return win_pnl / loss_pnl if loss_pnl > 0 else (win_pnl if win_pnl > 0 else 0.0)

    @property
    def be_rate(self) -> float:
        return self.n_breakeven / self.n_trades if self.n_trades > 0 else 0.0

    def next_trade_id(self) -> str:
        self._trade_counter += 1
        return f"{self.config.name}_{self._trade_counter}"


# ═══════════════════════════════════════════════════════════════════
#  Default 20 variants
# ═══════════════════════════════════════════════════════════════════

# Scalp-Day Hybrid variants (2026-04-19): no AAA tier dispatch, risk scales
# with alignment score. Variants explore threshold / RR / leverage space.
DEFAULT_VARIANTS: list[VariantConfig] = [
    # Baseline: threshold 0.78, RR 2.0, moderate leverage
    VariantConfig("Base-0.78-RR2",     0.78, 2.0,  5, 0.010),
    VariantConfig("Base-0.78-RR2-Lo",  0.78, 2.0,  5, 0.005),
    VariantConfig("Base-0.78-RR2-Hi",  0.78, 2.0,  5, 0.015),
    # Tighter threshold (fewer trades, higher conviction)
    VariantConfig("Tight-0.82-RR2",    0.82, 2.0,  5, 0.010),
    VariantConfig("Tight-0.85-RR2",    0.85, 2.0,  5, 0.010),
    VariantConfig("Tight-0.88-RR2.5",  0.88, 2.5,  5, 0.010),
    # RR sweeps (same gate, longer target)
    VariantConfig("Base-RR2.5",        0.78, 2.5,  5, 0.010),
    VariantConfig("Base-RR3.0",        0.78, 3.0,  5, 0.010),
    # Leverage sweeps
    VariantConfig("Base-Lev3",         0.78, 2.0,  3, 0.010),
    VariantConfig("Base-Lev7",         0.78, 2.0,  7, 0.010),
    VariantConfig("Base-Lev10",        0.78, 2.0, 10, 0.010),
    # Risk sweeps
    VariantConfig("Base-Risk0.3",      0.78, 2.0,  5, 0.003),
    VariantConfig("Base-Risk0.8",      0.78, 2.0,  5, 0.008),
    VariantConfig("Base-Risk1.5",      0.78, 2.0,  5, 0.015),
    # Tight + Lev combinations
    VariantConfig("Tight-0.85-Lev10",  0.85, 2.0, 10, 0.010),
    VariantConfig("Tight-0.88-Lev3",   0.88, 2.5,  3, 0.010),
    # Wild corners
    VariantConfig("Wild-Max-Agg",      0.78, 2.0, 10, 0.015),
    VariantConfig("Wild-Min-Defensive",0.90, 3.0,  3, 0.003),
    VariantConfig("Wild-HiRR",         0.80, 3.5,  5, 0.010),
    VariantConfig("Wild-LoRR",         0.78, 1.5,  5, 0.010),
]


# ═══════════════════════════════════════════════════════════════════
#  PaperGrid — main class
# ═══════════════════════════════════════════════════════════════════

class PaperGrid:
    """Evaluates signals against 20 parameter variants, tracks virtual PnL."""

    def __init__(
        self,
        variants: list[VariantConfig] | None = None,
        account_size: float = 100_000.0,
        results_dir: str = "paper_grid_results",
    ):
        # Load from variants.json if available and no explicit variants given
        if variants is None:
            variants = self._load_variants_from_file(results_dir)
        self.variants: list[VariantConfig] = variants or DEFAULT_VARIANTS
        self.states: dict[str, VariantState] = {
            v.name: VariantState(config=v, equity=account_size, peak_equity=account_size)
            for v in self.variants
        }
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._account_size = account_size
        # Count per-class variants
        class_counts: dict[str, int] = {}
        for v in self.variants:
            ac = v.asset_class or "global"
            class_counts[ac] = class_counts.get(ac, 0) + 1
        logger.info(
            "PaperGrid initialized: %d variants ($%.0f each) — %s",
            len(self.variants), account_size,
            ", ".join(f"{k}:{v}" for k, v in sorted(class_counts.items())),
        )

    @staticmethod
    def _load_variants_from_file(results_dir: str) -> list[VariantConfig] | None:
        """Load per-class variants from variants.json if it exists."""
        vfile = Path(results_dir) / "variants.json"
        if not vfile.exists():
            return None
        try:
            with open(vfile) as f:
                data = json.load(f)
            variants = []
            for d in data:
                variants.append(VariantConfig(
                    name=d["name"],
                    alignment_threshold=d["alignment_threshold"],
                    min_rr=d["min_rr"],
                    leverage=int(d["leverage"]),
                    risk_per_trade=d["risk_per_trade"],
                    asset_class=d.get("asset_class"),
                ))
            logger.info("Loaded %d variants from %s", len(variants), vfile)
            return variants
        except Exception as exc:
            logger.warning("Failed loading variants.json: %s — using defaults", exc)
            return None

    # ── Signal evaluation ───────────────────────────────────────

    def evaluate_signal(
        self,
        signal: dict[str, Any],
        asset_class: str = "crypto",
    ) -> dict[str, str | None]:
        """
        Evaluate a signal against all variants.

        Args:
            signal: dict with keys: symbol, direction, score, rr, sl, tp, entry_price, components
            asset_class: "crypto" | "forex" | "stocks" | "commodities"

        Returns:
            {variant_name: trade_id} for accepted, {variant_name: None} for rejected.
        """
        score = signal.get("score", 0.0)
        rr = signal.get("rr", 0.0)
        entry_price = signal.get("ref_price", signal.get("entry_price", 0.0))
        sl = signal.get("sl", 0.0)
        tp = signal.get("tp", 0.0)
        direction = signal.get("direction", "long")
        symbol = signal.get("symbol", "UNKNOWN")
        components = signal.get("components", {})

        results: dict[str, str | None] = {}
        n_accepted = 0

        for name, state in self.states.items():
            cfg = state.config

            # Filter 0: asset-class match (skip variants for other classes)
            if cfg.asset_class and cfg.asset_class != asset_class:
                results[name] = None
                continue

            # Filter 1: alignment threshold
            if score < cfg.alignment_threshold:
                results[name] = None
                continue

            # Filter 2: min RR
            if rr < cfg.min_rr:
                results[name] = None
                continue

            # Filter 3: max 1 open trade per variant (sniper approach)
            if state.open_trades:
                results[name] = None
                continue

            # ── Compute position size ───────────────────────────
            sl_dist = abs(entry_price - sl)
            if sl_dist <= 0 or entry_price <= 0:
                results[name] = None
                continue

            sl_pct = sl_dist / entry_price
            risk_pct = min(cfg.risk_per_trade, 0.03)  # hard cap 3%
            risk_amount = state.equity * risk_pct
            position_notional = risk_amount / sl_pct

            # Apply leverage cap
            max_notional = state.equity * cfg.leverage
            position_notional = min(position_notional, max_notional)

            # Fees
            commission = ASSET_COMMISSION.get(asset_class, 0.0004)
            cost = position_notional * (commission * 2 + SLIPPAGE_PCT * 2)

            # Create virtual trade
            trade_id = state.next_trade_id()
            vtrade = VirtualTrade(
                trade_id=trade_id,
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                stop_loss=sl,
                take_profit=tp,
                risk_reward=rr,
                position_notional=position_notional,
                risk_amount=risk_amount,
                cost=cost,
                asset_class=asset_class,
                entry_time=time.time(),
                score=score,
            )
            state.open_trades[trade_id] = vtrade
            results[name] = trade_id
            n_accepted += 1

        if n_accepted > 0:
            logger.info(
                "PaperGrid: %s %s evaluated → %d/%d variants accepted",
                symbol, direction, n_accepted, len(self.states),
            )

        return results

    # ── Trade close (called when real bot detects SL/TP/BE hit) ─

    def record_trade_close(
        self,
        exit_price: float,
        symbol: str,
    ) -> None:
        """
        Close all open virtual trades matching this symbol across all variants.
        Uses the actual exit price from the exchange.
        """
        for name, state in self.states.items():
            # Find matching open trade
            to_close = [
                tid for tid, vt in state.open_trades.items()
                if vt.symbol == symbol
            ]
            for trade_id in to_close:
                self._close_trade(state, trade_id, exit_price)

    def _close_trade(self, state: VariantState, trade_id: str, exit_price: float) -> None:
        """Close a specific virtual trade and update metrics."""
        vt = state.open_trades.pop(trade_id, None)
        if vt is None:
            return

        # PnL calculation (identical to backtester)
        if vt.direction == "long":
            price_pnl_pct = (exit_price - vt.entry_price) / vt.entry_price
        else:
            price_pnl_pct = (vt.entry_price - exit_price) / vt.entry_price

        gross_pnl = vt.position_notional * price_pnl_pct
        net_pnl = gross_pnl - vt.cost

        # Classify outcome
        sl_dist = abs(vt.entry_price - vt.stop_loss)
        if sl_dist > 0:
            actual_rr = (net_pnl / vt.risk_amount) if vt.risk_amount > 0 else 0.0
        else:
            actual_rr = 0.0

        if actual_rr > 0.2:
            outcome = "win"
            state.n_wins += 1
        elif actual_rr < -0.2:
            outcome = "loss"
            state.n_losses += 1
        else:
            outcome = "breakeven"
            state.n_breakeven += 1

        # Update equity
        state.equity += net_pnl
        state.total_pnl += net_pnl
        if state.equity > state.peak_equity:
            state.peak_equity = state.equity

        # Record trade
        trade_record = {
            "trade_id": trade_id,
            "symbol": vt.symbol,
            "direction": vt.direction,
            "entry_price": vt.entry_price,
            "exit_price": exit_price,
            "pnl": net_pnl,
            "actual_rr": actual_rr,
            "outcome": outcome,
            "score": vt.score,
            "cost": vt.cost,
            "entry_time": vt.entry_time,
            "exit_time": time.time(),
        }
        state.trades.append(trade_record)

        logger.info(
            "PaperGrid [%s]: %s %s %s → %s PnL=$%.2f RR=%.2f Equity=$%.0f",
            state.config.name, outcome.upper(), vt.direction, vt.symbol,
            exit_price, net_pnl, actual_rr, state.equity,
        )

    # ── Dashboard data ──────────────────────────────────────────

    def dashboard_data(self) -> list[dict[str, Any]]:
        """Return sorted list of variant summaries for Rich dashboard."""
        rows = []
        for name, state in self.states.items():
            rows.append({
                "name": name,
                "equity": state.equity,
                "pnl": state.total_pnl,
                "pnl_pct": (state.total_pnl / self._account_size) * 100,
                "dd_pct": state.drawdown_pct * 100,
                "trades": state.n_trades,
                "wins": state.n_wins,
                "losses": state.n_losses,
                "be": state.n_breakeven,
                "wr_real": state.winrate_real * 100,
                "pf_real": state.pf_real,
                "be_rate": state.be_rate * 100,
                "open": len(state.open_trades),
                "align": state.config.alignment_threshold,
                "rr": state.config.min_rr,
                "lev": state.config.leverage,
                "risk": state.config.risk_per_trade * 100,
            })
        # Sort by PnL descending
        rows.sort(key=lambda r: r["pnl"], reverse=True)
        return rows

    # ── Export ───────────────────────────────────────────────────

    def export_csv(self) -> Path:
        """Export all variant trade histories to CSV."""
        out = self.results_dir / "paper_grid_trades.csv"
        all_trades = []
        for name, state in self.states.items():
            for t in state.trades:
                row = {"variant": name, **t}
                all_trades.append(row)

        if not all_trades:
            return out

        fieldnames = list(all_trades[0].keys())
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_trades)

        logger.info("PaperGrid exported %d trades → %s", len(all_trades), out)
        return out

    def export_summary(self) -> Path:
        """Export variant summary to CSV."""
        out = self.results_dir / "paper_grid_summary.csv"
        rows = self.dashboard_data()
        if not rows:
            return out

        fieldnames = list(rows[0].keys())
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return out

    # ── Persistence (save/load state) ───────────────────────────

    def save_state(self) -> None:
        """Save grid state to JSON for crash recovery."""
        state_file = self.results_dir / "paper_grid_state.json"
        data = {}
        for name, state in self.states.items():
            data[name] = {
                "equity": state.equity,
                "peak_equity": state.peak_equity,
                "total_pnl": state.total_pnl,
                "n_wins": state.n_wins,
                "n_losses": state.n_losses,
                "n_breakeven": state.n_breakeven,
                "trades": state.trades,
                "_trade_counter": state._trade_counter,
            }
        with open(state_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load_state(self) -> bool:
        """Load grid state from JSON. Returns True if loaded."""
        state_file = self.results_dir / "paper_grid_state.json"
        if not state_file.exists():
            return False
        try:
            with open(state_file) as f:
                data = json.load(f)
            for name, saved in data.items():
                if name in self.states:
                    s = self.states[name]
                    s.equity = saved["equity"]
                    s.peak_equity = saved["peak_equity"]
                    s.total_pnl = saved["total_pnl"]
                    s.n_wins = saved["n_wins"]
                    s.n_losses = saved["n_losses"]
                    s.n_breakeven = saved["n_breakeven"]
                    s.trades = saved["trades"]
                    s._trade_counter = saved.get("_trade_counter", len(saved["trades"]))
            logger.info("PaperGrid state loaded from %s", state_file)
            return True
        except Exception as exc:
            logger.warning("Failed to load PaperGrid state: %s", exc)
            return False
