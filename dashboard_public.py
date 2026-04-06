"""
Public Trading Dashboard -- read-only, no auth required.
Runs on port 8081 alongside the admin dashboard on 8080.

Reads from:
  - trade_journal/journal.db  (SQLite WAL, read-only)
  - paper_trading.log         (bot log file)
  - live_results/             (bot state JSONs, equity CSVs)
  - /proc or shutil           (system stats)

Usage: python3 dashboard_public.py [--port 8081]
"""
from __future__ import annotations

import csv
import json
import os
import platform
import re
import shutil
import sqlite3
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

RESULTS_DIR = Path("live_results")
DB_PATH = Path("trade_journal/journal.db")
LOG_PATH = Path("paper_trading.log")
LIVE_LOG_PATH = Path("live_results/live_multi.log")
MODEL_DIR = Path("models")

# Display multipliers for asset classes with virtual equity (e.g., Binance testnet 5K → 100K)
try:
    import yaml as _yaml
    _cfg = _yaml.safe_load(open("config/default_config.yaml"))
    DISPLAY_MULTIPLIERS: dict[str, float] = {
        k: float(v) for k, v in _cfg.get("equity_display_multipliers", {}).items()
    }
except Exception:
    DISPLAY_MULTIPLIERS = {}

def _apply_display_mult(ac: str, value: float) -> float:
    """Multiply value by display multiplier for the asset class (if configured)."""
    return value * DISPLAY_MULTIPLIERS.get(ac, 1.0)

# =====================================================================
#  Helpers
# =====================================================================


def _query_journal(sql: str, params: tuple = ()) -> list[dict]:
    """Execute a read-only query against the journal DB."""
    if not DB_PATH.exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _journal_columns() -> set[str]:
    """Return column names in the trades table, or empty set."""
    if not DB_PATH.exists():
        return set()
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(trades)")
        cols = {row[1] for row in cur.fetchall()}
        conn.close()
        return cols
    except Exception:
        return set()


def _journal_has_table(name: str) -> bool:
    if not DB_PATH.exists():
        return False
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        )
        found = cur.fetchone() is not None
        conn.close()
        return found
    except Exception:
        return False


def _find_bot_pid() -> int | None:
    """Find PID of live_multi_bot.py process."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "live_multi_bot.py"],
            capture_output=True, text=True, timeout=5,
        )
        pids = result.stdout.strip().split("\n")
        pids = [p.strip() for p in pids if p.strip()]
        if pids:
            return int(pids[0])
    except Exception:
        pass
    return None


def _get_bot_uptime() -> str:
    """Get bot uptime from paper_trading.log first line timestamp."""
    if not LOG_PATH.exists():
        return "unknown"
    try:
        with open(LOG_PATH) as f:
            first_line = f.readline()
        m = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", first_line)
        if m:
            start = datetime.fromisoformat(m.group(1)).replace(tzinfo=timezone.utc)
            delta = datetime.now(timezone.utc) - start
            total_s = int(delta.total_seconds())
            if total_s < 0:
                return "0h 0m"
            days = total_s // 86400
            hours = (total_s % 86400) // 3600
            mins = (total_s % 3600) // 60
            if days > 0:
                return f"{days}d {hours}h {mins}m"
            return f"{hours}h {mins}m"
    except Exception:
        pass
    return "unknown"


def _current_session() -> str:
    """Determine current market session based on UTC hour."""
    hour = datetime.now(timezone.utc).hour
    if 0 <= hour < 7:
        return "Asian"
    if 7 <= hour < 12:
        return "London"
    if 12 <= hour < 17:
        return "London/NY Overlap"
    if 17 <= hour < 21:
        return "New York"
    return "After Hours"


def _read_log_tail(n: int = 50) -> list[str]:
    """Return last n lines from paper_trading.log."""
    if not LOG_PATH.exists():
        return []
    try:
        with open(LOG_PATH, "rb") as f:
            # Seek from end for efficiency
            try:
                f.seek(0, 2)
                fsize = f.tell()
                # Read last ~64KB max
                chunk = min(fsize, 65536)
                f.seek(max(fsize - chunk, 0))
                data = f.read().decode("utf-8", errors="replace")
            except Exception:
                f.seek(0)
                data = f.read().decode("utf-8", errors="replace")
        lines = data.strip().split("\n")
        return lines[-n:]
    except Exception:
        return []


def _get_near_misses(n: int = 50) -> list[dict]:
    """Extract last n NEAR-MISS entries as structured objects."""
    if not LOG_PATH.exists():
        return []

    _PAT_ALIGNMENT = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*NEAR-MISS ALIGNMENT (\S+) \| class=(\w+) score=([\d.]+) thresh=([\d.]+) dir=(\w+)"
    )
    _PAT_XGB = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*NEAR-MISS XGB (\S+) conf=([\d.]+) score=([\d.]+) thresh=([\d.]+) \| (\w+) (\w+)"
    )
    _PAT_NEUTRAL = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*NEAR-MISS NEUTRAL-BIAS (\S+) \| class=(\w+)"
    )

    hits: list[dict] = []
    try:
        with open(LOG_PATH) as f:
            for line in f:
                if "NEAR-MISS" not in line:
                    continue

                m = _PAT_ALIGNMENT.search(line)
                if m:
                    score_val = float(m.group(4))
                    thresh_val = float(m.group(5))
                    hits.append({
                        "time": m.group(1),
                        "symbol": m.group(2),
                        "type": "ALIGNMENT",
                        "score": round(score_val, 3),
                        "threshold": round(thresh_val, 2),
                        "gap": round(thresh_val - score_val, 3),
                        "direction": m.group(6),
                        "asset_class": m.group(3),
                    })
                    continue

                m = _PAT_XGB.search(line)
                if m:
                    conf_val = float(m.group(3))
                    thresh_val = float(m.group(5))
                    hits.append({
                        "time": m.group(1),
                        "symbol": m.group(2),
                        "type": "XGB",
                        "score": round(conf_val, 3),
                        "threshold": round(thresh_val, 2),
                        "gap": round(thresh_val - conf_val, 3),
                        "direction": m.group(7),
                        "asset_class": m.group(6),
                        "alignment_score": round(float(m.group(4)), 2),
                    })
                    continue

                m = _PAT_NEUTRAL.search(line)
                if m:
                    hits.append({
                        "time": m.group(1),
                        "symbol": m.group(2),
                        "type": "NEUTRAL_BIAS",
                        "score": 0.0,
                        "threshold": 0.0,
                        "gap": 0.0,
                        "direction": "none",
                        "asset_class": m.group(3),
                    })
                    continue

                # Fallback for unrecognized near-miss formats
                ts_match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
                hits.append({
                    "time": ts_match.group(1) if ts_match else "",
                    "symbol": "",
                    "type": "OTHER",
                    "score": 0.0,
                    "threshold": 0.0,
                    "gap": 0.0,
                    "direction": "",
                    "asset_class": "",
                    "raw": line.strip(),
                })
        return hits[-n:]
    except Exception:
        return []


def _discover_bots() -> dict[str, dict]:
    """Discover all bots from state files in live_results/."""
    bots: dict[str, dict] = {}
    if not RESULTS_DIR.exists():
        return bots
    for p in sorted(RESULTS_DIR.glob("bot_*_state.json")):
        tag = p.stem.replace("_state", "")
        try:
            with open(p) as f:
                state = json.load(f)
            # Infer asset class and symbol from per-bot log
            log_path = RESULTS_DIR / f"{tag}.log"
            ac = "unknown"
            symbol = tag
            if log_path.exists():
                with open(log_path) as lf:
                    for line in lf:
                        m = re.search(r"class=(\w+)", line)
                        if m:
                            ac = m.group(1)
                        m2 = re.search(r"symbol=([^\s|]+)", line)
                        if m2:
                            symbol = m2.group(1)
                        if ac != "unknown" and symbol != tag:
                            break
            bots[tag] = {
                "tag": tag,
                "symbol": symbol,
                "asset_class": ac,
                **state,
            }
        except Exception:
            continue
    return bots


def _aggregate_stats() -> dict:
    """Aggregate stats across all bots from live_results state files.

    Per-broker DD, active trades, and heat are computed individually
    because each broker maps to a separate funded account (100K each).
    """
    bots = _discover_bots()
    total_trades = 0
    total_wins = 0
    total_pnl = 0.0
    total_equity = 0.0
    active_positions = 0
    per_class: dict[str, dict] = {}

    # Per-broker: sum PnL from ALL bots, equity = initial + total PnL
    # OANDA serves both forex AND commodities from one 100K account.
    _BROKER_MAP = {"crypto": "binance", "forex": "oanda", "stocks": "alpaca", "commodities": "oanda"}
    _INITIAL_EQUITY = 100_000.0  # all brokers normalized to 100K display
    broker_pnl: dict[str, float] = {}  # sum of display PnL per broker
    broker_active: dict[str, int] = {}  # open trades per broker
    broker_heat: dict[str, float] = {}  # portfolio heat per broker

    for tag, bot in bots.items():
        trades = bot.get("trades", 0)
        wins = bot.get("wins", 0)
        pnl = bot.get("total_pnl", 0.0)
        actives = bot.get("active_trades", [])

        ac = bot.get("asset_class", "unknown")
        disp_pnl = _apply_display_mult(ac, pnl)
        broker = _BROKER_MAP.get(ac, ac)

        total_trades += trades
        total_wins += wins
        total_pnl += disp_pnl
        active_positions += len(actives)

        if ac not in per_class:
            per_class[ac] = {"trades": 0, "wins": 0, "pnl": 0.0, "equity": 0.0,
                             "active": 0}
        per_class[ac]["trades"] += trades
        per_class[ac]["wins"] += wins
        per_class[ac]["pnl"] += disp_pnl
        per_class[ac]["active"] += len(actives)

        # Accumulate per broker
        broker_pnl[broker] = broker_pnl.get(broker, 0.0) + disp_pnl
        broker_active[broker] = broker_active.get(broker, 0) + len(actives)

        # Heat per broker (non-BE trades only) — risk_pct stored as fraction (0.01 = 1%)
        for trade in actives:
            if isinstance(trade, dict) and not trade.get("be_triggered", False):
                broker_heat[broker] = broker_heat.get(broker, 0.0) + float(trade.get("risk_pct", 0.0)) * 100

    # Compute equity + DD per broker (each is a separate funded account)
    per_broker: dict[str, dict] = {}
    worst_dd_pct = 0.0
    for broker, pnl_sum in broker_pnl.items():
        broker_equity = _INITIAL_EQUITY + pnl_sum
        total_equity += broker_equity
        dd_pct = max(0.0, (_INITIAL_EQUITY - broker_equity) / _INITIAL_EQUITY * 100)
        worst_dd_pct = max(worst_dd_pct, dd_pct)
        per_broker[broker] = {
            "equity": round(broker_equity, 2),
            "pnl": round(pnl_sum, 2),
            "pnl_pct": round(pnl_sum / _INITIAL_EQUITY * 100, 2),
            "dd_pct": round(dd_pct, 2),
            "active": broker_active.get(broker, 0),
            "heat_pct": round(broker_heat.get(broker, 0.0), 2),
        }
        # Map back to per-class for display
        for ac, br in _BROKER_MAP.items():
            if br == broker and ac in per_class:
                per_class[ac]["equity"] = round(broker_equity, 2)

    wr = round(total_wins / total_trades * 100, 2) if total_trades > 0 else 0.0

    return {
        "total_trades": total_trades,
        "total_wins": total_wins,
        "win_rate": wr,
        "total_pnl": round(total_pnl, 4),
        "total_pnl_pct": round(total_pnl / (_INITIAL_EQUITY * max(len(broker_pnl), 1)) * 100, 2),
        "total_equity": round(total_equity, 2),
        "active_positions": active_positions,
        "max_dd_pct": round(worst_dd_pct, 2),
        "per_class": per_class,
        "per_broker": per_broker,
    }


def _get_rl_stats() -> dict:
    """Check model files on disk and parse logs for RL/XGB filter stats."""
    stats: dict = {
        "entry_filter": {"accepted": 0, "rejected": 0, "rate": 0.0},
        "tp_adjusted": 0,
        "be_triggered": 0,
        "models_loaded": [],
    }

    # Count loaded models by checking which .pkl files exist
    if MODEL_DIR.is_dir():
        for pkl in sorted(MODEL_DIR.glob("rl_*.pkl")):
            try:
                stats["models_loaded"].append(pkl.name)
            except Exception:
                pass
        # Also check DQN
        if (MODEL_DIR / "dqn_exit_manager.zip").exists():
            stats["models_loaded"].append("dqn_exit_manager.zip")

    # Parse BOTH log files for XGB ACCEPT/REJECT, TP adjusted, BE TRIGGERED
    for log_path in (LOG_PATH, LIVE_LOG_PATH):
        if not log_path.exists():
            continue
        try:
            with open(log_path) as f:
                for line in f:
                    if "XGB ACCEPT" in line:
                        stats["entry_filter"]["accepted"] += 1
                    elif "XGB REJECT" in line:
                        stats["entry_filter"]["rejected"] += 1
                    elif "RL TP adjusted" in line:
                        stats["tp_adjusted"] += 1
                    elif "BE TRIGGERED" in line:
                        stats["be_triggered"] += 1
        except Exception:
            pass

    total = stats["entry_filter"]["accepted"] + stats["entry_filter"]["rejected"]
    if total > 0:
        stats["entry_filter"]["rate"] = round(
            stats["entry_filter"]["accepted"] / total * 100, 1
        )
    return stats


def _get_equity_curve() -> list[dict]:
    """Build portfolio equity curve from journal closed trades.

    Starts at 300K (3 brokers × 100K), adds each trade's PnL chronologically.
    PnL is display-adjusted (crypto ×20).
    """
    _INITIAL_PORTFOLIO = 300_000.0  # 3 brokers × 100K

    if not _journal_has_table("trades"):
        return [{"timestamp": "", "equity": _INITIAL_PORTFOLIO, "pnl": 0.0}]

    rows = _query_journal(
        "SELECT exit_time, pnl_pct, asset_class FROM trades "
        "WHERE exit_time IS NOT NULL ORDER BY exit_time ASC"
    )

    if not rows:
        return [{"timestamp": "", "equity": _INITIAL_PORTFOLIO, "pnl": 0.0}]

    # Each trade's $ PnL = pnl_pct × 100K (all accounts normalized to 100K display)
    cumulative_pnl = 0.0
    points = [{"timestamp": rows[0].get("exit_time", "")[:16], "equity": _INITIAL_PORTFOLIO, "pnl": 0.0}]

    for r in rows:
        pnl_dollar = (r.get("pnl_pct") or 0.0) * 100_000.0
        cumulative_pnl += pnl_dollar
        points.append({
            "timestamp": (r.get("exit_time") or "")[:16],
            "equity": round(_INITIAL_PORTFOLIO + cumulative_pnl, 2),
            "pnl": round(cumulative_pnl, 2),
        })

    return points[-500:]


def _system_stats() -> dict:
    """Gather system stats (CPU, RAM, disk) without psutil."""
    stats: dict = {"cpu_pct": 0.0, "ram_total_mb": 0, "ram_used_mb": 0,
                   "ram_pct": 0.0, "disk_total_gb": 0, "disk_used_gb": 0,
                   "disk_pct": 0.0, "bot_rss_mb": 0.0}

    is_linux = platform.system() == "Linux"

    # CPU load
    try:
        if is_linux:
            with open("/proc/loadavg") as f:
                load1 = float(f.read().split()[0])
            cpu_count = os.cpu_count() or 1
            stats["cpu_pct"] = round(min(load1 / cpu_count * 100, 100), 1)
        else:
            result = subprocess.run(
                ["sysctl", "-n", "vm.loadavg"],
                capture_output=True, text=True, timeout=3,
            )
            parts = result.stdout.strip().strip("{}").split()
            if parts:
                load1 = float(parts[0])
                cpu_count = os.cpu_count() or 1
                stats["cpu_pct"] = round(min(load1 / cpu_count * 100, 100), 1)
    except Exception:
        pass

    # RAM
    try:
        if is_linux:
            with open("/proc/meminfo") as f:
                meminfo = f.read()
            total_match = re.search(r"MemTotal:\s+(\d+)", meminfo)
            avail_match = re.search(r"MemAvailable:\s+(\d+)", meminfo)
            if total_match and avail_match:
                total_kb = int(total_match.group(1))
                avail_kb = int(avail_match.group(1))
                used_kb = total_kb - avail_kb
                stats["ram_total_mb"] = total_kb // 1024
                stats["ram_used_mb"] = used_kb // 1024
                stats["ram_pct"] = round(used_kb / total_kb * 100, 1)
        else:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=3,
            )
            total_bytes = int(result.stdout.strip())
            stats["ram_total_mb"] = total_bytes // (1024 * 1024)
            result2 = subprocess.run(
                ["vm_stat"], capture_output=True, text=True, timeout=3,
            )
            free_match = re.search(r"Pages free:\s+(\d+)", result2.stdout)
            inactive_match = re.search(r"Pages inactive:\s+(\d+)", result2.stdout)
            free_pages = int(free_match.group(1)) if free_match else 0
            inactive_pages = int(inactive_match.group(1)) if inactive_match else 0
            avail_bytes = (free_pages + inactive_pages) * 4096
            used_bytes = total_bytes - avail_bytes
            stats["ram_used_mb"] = max(0, used_bytes // (1024 * 1024))
            stats["ram_pct"] = round(
                max(0, min(100, used_bytes / total_bytes * 100)), 1
            )
    except Exception:
        pass

    # Disk
    try:
        usage = shutil.disk_usage("/")
        stats["disk_total_gb"] = round(usage.total / (1024**3), 1)
        stats["disk_used_gb"] = round(usage.used / (1024**3), 1)
        stats["disk_pct"] = round(usage.used / usage.total * 100, 1)
    except Exception:
        pass

    # Bot process RSS
    pid = _find_bot_pid()
    if pid:
        try:
            if is_linux:
                with open(f"/proc/{pid}/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            rss_kb = int(line.split()[1])
                            stats["bot_rss_mb"] = round(rss_kb / 1024, 1)
                            break
            else:
                result = subprocess.run(
                    ["ps", "-o", "rss=", "-p", str(pid)],
                    capture_output=True, text=True, timeout=3,
                )
                rss_kb = int(result.stdout.strip())
                stats["bot_rss_mb"] = round(rss_kb / 1024, 1)
        except Exception:
            pass

    return stats


def _get_open_positions() -> list[dict]:
    """Collect currently open positions from bot state files."""
    positions: list[dict] = []
    bots = _discover_bots()
    for tag, bot in bots.items():
        ac = bot.get("asset_class", "unknown")
        mult = DISPLAY_MULTIPLIERS.get(ac, 1.0)
        for trade in bot.get("active_trades", []):
            pos = {
                "symbol": bot.get("symbol", tag),
                "asset_class": ac,
                "bot": tag,
            }
            if isinstance(trade, dict):
                pos.update(trade)
                # Apply display multiplier to PnL fields for crypto
                if mult > 1:
                    for pnl_key in ("unrealized_pnl_usd", "pnl_usd", "risk_amount"):
                        if pnl_key in pos and isinstance(pos[pnl_key], (int, float)):
                            pos[pnl_key] = pos[pnl_key] * mult
            else:
                pos["info"] = str(trade)
            positions.append(pos)
    return positions


def _get_risk_status() -> dict:
    """Read circuit breaker limits and approximate current status."""
    risk = {
        "daily_loss_limit": 3.0,
        "weekly_loss_limit": 5.0,
        "alltime_dd_limit": 8.0,
        "max_heat": 6.0,
        "daily_pnl_pct": 0.0,
        "weekly_pnl_pct": 0.0,
        "alltime_dd_pct": 0.0,
        "portfolio_heat_pct": 0.0,
        "daily_breaker": False,
        "weekly_breaker": False,
        "alltime_breaker": False,
        "heat_breaker": False,
        "asset_class_paused": {},
    }
    # Parse log for breaker activations
    if not LOG_PATH.exists():
        return risk
    try:
        lines = _read_log_tail(500)
        for line in lines:
            if "DAILY BREAKER" in line.upper() or "daily_breaker_active" in line:
                risk["daily_breaker"] = True
            if "WEEKLY BREAKER" in line.upper() or "weekly_breaker_active" in line:
                risk["weekly_breaker"] = True
            if "ALLTIME" in line.upper() and "BREAKER" in line.upper():
                risk["alltime_breaker"] = True
            if "HEAT BREAKER" in line.upper() or "heat_breaker" in line:
                risk["heat_breaker"] = True
            m_daily = re.search(r"daily_pnl[=:]\s*([-\d.]+)%?", line, re.IGNORECASE)
            if m_daily:
                risk["daily_pnl_pct"] = float(m_daily.group(1))
            m_weekly = re.search(r"weekly_pnl[=:]\s*([-\d.]+)%?", line, re.IGNORECASE)
            if m_weekly:
                risk["weekly_pnl_pct"] = float(m_weekly.group(1))
            m_heat = re.search(r"heat[=:]\s*([-\d.]+)%?", line, re.IGNORECASE)
            if m_heat:
                risk["portfolio_heat_pct"] = float(m_heat.group(1))
    except Exception:
        pass

    # Approximate DD from state files
    agg = _aggregate_stats()
    risk["alltime_dd_pct"] = agg.get("max_dd_pct", 0.0)

    return risk


# =====================================================================
#  API Endpoints (all GET, no auth)
# =====================================================================


@app.route("/api/public/status")
def api_status():
    pid = _find_bot_pid()
    return jsonify({
        "running": pid is not None,
        "pid": pid,
        "uptime": _get_bot_uptime(),
        "session": _current_session(),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    })


@app.route("/api/public/overview")
def api_overview():
    agg = _aggregate_stats()
    # Compute profit factor from journal if available
    pf = 0.0
    sharpe = 0.0
    if _journal_has_table("trades"):
        rows = _query_journal(
            "SELECT pnl_pct FROM trades WHERE exit_time IS NOT NULL AND pnl_pct IS NOT NULL"
        )
        gains = sum(r["pnl_pct"] for r in rows if r["pnl_pct"] > 0)
        losses = abs(sum(r["pnl_pct"] for r in rows if r["pnl_pct"] < 0))
        if losses > 0:
            pf = round(gains / losses, 2)
        elif gains > 0:
            pf = 999.0
        # Sharpe approximation
        if len(rows) > 1:
            returns = [r["pnl_pct"] for r in rows]
            mean_r = sum(returns) / len(returns)
            var_r = sum((x - mean_r) ** 2 for x in returns) / (len(returns) - 1)
            std_r = var_r ** 0.5
            if std_r > 0:
                sharpe = round(mean_r / std_r * (252 ** 0.5), 2)

    return jsonify({
        **agg,
        "profit_factor": pf,
        "sharpe": sharpe,
    })


@app.route("/api/public/equity")
def api_equity():
    return jsonify(_get_equity_curve())


@app.route("/api/public/trades")
def api_trades():
    if not _journal_has_table("trades"):
        return jsonify([])
    desired = [
        "trade_id", "symbol", "asset_class", "direction",
        "entry_time", "exit_time", "entry_price", "exit_price",
        "pnl_pct", "rr_actual", "rr_target", "outcome",
        "exit_reason", "bars_held", "tier", "score",
    ]
    existing = _journal_columns()
    if not existing:
        return jsonify([])
    cols = [c for c in desired if c in existing]
    if not cols:
        return jsonify([])
    col_sql = ", ".join(cols)
    rows = _query_journal(
        f"SELECT {col_sql} FROM trades WHERE exit_time IS NOT NULL "
        "ORDER BY exit_time DESC LIMIT 100"
    )
    return jsonify(rows)


@app.route("/api/public/positions")
def api_positions():
    return jsonify(_get_open_positions())


@app.route("/api/public/per-class")
def api_per_class():
    agg = _aggregate_stats()
    per_class = agg.get("per_class", {})
    # Add win rate per class
    for ac, data in per_class.items():
        t = data.get("trades", 0)
        w = data.get("wins", 0)
        data["win_rate"] = round(w / t * 100, 1) if t > 0 else 0.0
    return jsonify(per_class)


@app.route("/api/public/period-stats")
def api_period_stats():
    """Return daily/weekly/monthly PnL and DD from journal trades."""
    if not _journal_has_table("trades"):
        return jsonify({"daily": {}, "weekly": {}, "monthly": {}})
    cols = _journal_columns()
    has_ac = "asset_class" in cols

    def _period(where_clause: str) -> dict:
        sql = (
            "SELECT pnl_pct, outcome"
            + (", asset_class" if has_ac else "")
            + " FROM trades WHERE exit_time IS NOT NULL AND " + where_clause
        )
        rows = _query_journal(sql)
        if not rows:
            return {"pnl_pct": 0.0, "pnl_dollar": 0.0, "trades": 0, "wins": 0, "losses": 0, "dd_pct": 0.0}
        total_pct = sum(r["pnl_pct"] or 0.0 for r in rows)
        wins = sum(1 for r in rows if r["outcome"] == "win")
        losses = len(rows) - wins
        # DD = worst running drawdown within the period
        running = 0.0
        peak = 0.0
        worst_dd = 0.0
        for r in rows:
            running += r["pnl_pct"] or 0.0
            if running > peak:
                peak = running
            dd = peak - running
            if dd > worst_dd:
                worst_dd = dd
        return {
            "pnl_pct": round(total_pct * 100, 2),
            "pnl_dollar": round(total_pct * 100_000.0, 2),
            "trades": len(rows),
            "wins": wins,
            "losses": losses,
            "dd_pct": round(worst_dd * 100, 2),
        }

    daily = _period("DATE(exit_time) = DATE('now')")
    weekly = _period("DATE(exit_time) >= DATE('now', '-7 days')")
    monthly = _period("DATE(exit_time) >= DATE('now', '-30 days')")

    return jsonify({"daily": daily, "weekly": weekly, "monthly": monthly})


@app.route("/api/public/rl-stats")
def api_rl_stats():
    stats = _get_rl_stats()
    stats["xgb_accepted"] = stats["entry_filter"]["accepted"]
    stats["xgb_rejected"] = stats["entry_filter"]["rejected"]
    return jsonify(stats)


@app.route("/api/public/near-misses")
def api_near_misses():
    return jsonify(_get_near_misses(50))


@app.route("/api/public/risk")
def api_risk():
    return jsonify(_get_risk_status())


@app.route("/api/public/circuit-breaker")
def api_circuit_breaker():
    """Return per-broker circuit breaker status (each broker = funded account)."""
    _BROKER_MAP = {"crypto": "binance", "forex": "oanda", "stocks": "alpaca", "commodities": "oanda"}
    _INITIAL_EQUITY = 100_000.0

    # Per-broker PnL + heat from bot state files
    agg = _aggregate_stats()
    per_broker = agg.get("per_broker", {})

    # Per-broker daily/weekly PnL from journal
    has_ac = "asset_class" in _journal_columns() if _journal_has_table("trades") else False
    if has_ac:
        for period, where in [("daily", "DATE(exit_time) = DATE('now')"),
                               ("weekly", "DATE(exit_time) >= DATE('now', '-7 days')")]:
            rows = _query_journal(
                f"SELECT asset_class, pnl_pct FROM trades WHERE exit_time IS NOT NULL AND {where}"
            )
            broker_period: dict[str, float] = {}
            for r in rows:
                ac = r.get("asset_class", "unknown")
                broker = _BROKER_MAP.get(ac, ac)
                broker_period[broker] = broker_period.get(broker, 0.0) + (r["pnl_pct"] or 0.0)
            for broker in per_broker:
                per_broker[broker][f"{period}_pnl_pct"] = round(broker_period.get(broker, 0.0) * 100, 2)

    # Compute per-broker status and risk budget
    any_breaker = False
    for broker, bd in per_broker.items():
        daily = bd.get("daily_pnl_pct", 0.0)
        weekly = bd.get("weekly_pnl_pct", 0.0)
        alltime_dd = bd.get("dd_pct", 0.0)
        heat = bd.get("heat_pct", 0.0)

        # Breaker flags per broker
        daily_br = daily <= -3.0
        weekly_br = weekly <= -5.0
        alltime_br = alltime_dd >= 8.0
        heat_br = heat >= 6.0

        if daily_br:
            bd["status"] = "DAILY_PAUSE"
        elif weekly_br:
            bd["status"] = "WEEKLY_PAUSE"
        elif alltime_br:
            bd["status"] = "ALLTIME_STOP"
        elif heat_br:
            bd["status"] = "HEAT_PAUSE"
        else:
            bd["status"] = "CLEAR"

        bd["daily_breaker"] = daily_br
        bd["weekly_breaker"] = weekly_br
        bd["alltime_breaker"] = alltime_br
        bd["heat_breaker"] = heat_br
        bd["risk_budget"] = round(max(0.0, 3.0 - abs(min(daily, 0)) - heat), 2)

        if daily_br or weekly_br or alltime_br or heat_br:
            any_breaker = True

    return jsonify({
        "per_broker": per_broker,
        "any_breaker": any_breaker,
        "daily_loss_limit": 3.0,
        "weekly_loss_limit": 5.0,
        "alltime_dd_limit": 8.0,
    })


@app.route("/api/public/logs")
def api_logs():
    return jsonify(_read_log_tail(50))


@app.route("/api/public/system")
def api_system():
    return jsonify(_system_stats())


def _format_age(seconds: int) -> str:
    """Human-readable age string."""
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m ago"
    return f"{seconds // 86400}d ago"


@app.route("/api/public/connections")
def api_connections():
    """Per-exchange connection status + last candle timestamps."""
    log_path = Path("paper_trading.log")
    heartbeat_path = Path("live_results/heartbeat.json")
    exchanges = {
        "Binance": {"status": "unknown", "last_data": None, "last_error": None, "asset_class": "crypto"},
        "OANDA": {"status": "unknown", "last_data": None, "last_error": None, "asset_class": "forex"},
        "Alpaca": {"status": "unknown", "last_data": None, "last_error": None, "asset_class": "stocks"},
    }
    per_class = {}
    for ac in ["crypto", "forex", "stocks", "commodities"]:
        per_class[ac] = {"last_candle": None, "stale": True, "connected": False}

    if not log_path.exists():
        return jsonify({"exchanges": exchanges, "per_class": per_class})

    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
    except Exception:
        return jsonify({"exchanges": exchanges, "per_class": per_class})

    now = datetime.now(timezone.utc)

    for line in lines:
        ts_match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
        ts_str = ts_match.group(1) if ts_match else None

        # Exchange connection status
        if "Binance (crypto): connected" in line:
            exchanges["Binance"]["status"] = "connected"
        if "OANDA (forex+commodities): connected" in line:
            exchanges["OANDA"]["status"] = "connected"
        if "Alpaca (stocks): connected" in line or "AlpacaAdapter connected" in line:
            exchanges["Alpaca"]["status"] = "connected"

        # Track errors per exchange
        if "ERROR" in line and ts_str:
            if any(fx in line for fx in ["EUR_", "GBP_", "USD_", "AUD_", "NZD_", "CAD_", "CHF_"]):
                exchanges["OANDA"]["last_error"] = ts_str
            elif "XAU_" in line or "XAG_" in line or "WTICO_" in line or "BCO_" in line:
                exchanges["OANDA"]["last_error"] = ts_str
            elif "USDT" in line:
                exchanges["Binance"]["last_error"] = ts_str

        # Track last candle per class (from NEAR-MISS or Loaded lines)
        if ts_str and ("NEAR-MISS" in line or "Loaded" in line or "on_candle" in line):
            for ac in ["crypto", "forex", "stocks", "commodities"]:
                if f"class={ac}" in line:
                    per_class[ac]["last_candle"] = ts_str
                    per_class[ac]["connected"] = True
            # Infer from symbol patterns
            if any(s in line for s in ["USDT", "/USDT"]):
                per_class["crypto"]["last_candle"] = ts_str
                per_class["crypto"]["connected"] = True
            if any(s in line for s in ["EUR_", "GBP_", "USD_J", "AUD_", "NZD_", "CAD_", "CHF_"]):
                per_class["forex"]["last_candle"] = ts_str
                per_class["forex"]["connected"] = True
            if any(s in line for s in ["XAU_", "XAG_", "WTICO_", "BCO_"]):
                per_class["commodities"]["last_candle"] = ts_str
                per_class["commodities"]["connected"] = True

        # HEARTBEAT entries prove the bot is alive and processing candles
        if ts_str and "HEARTBEAT:" in line:
            # HEARTBEAT updates freshness for ALL classes that report candles > 0
            # Format: HEARTBEAT: candles_5m=[crypto=X forex=Y ...] ...
            for ac in ["crypto", "forex", "stocks", "commodities"]:
                # Check if this class had candles (e.g. "crypto=42" means active)
                m = re.search(rf"{ac}=(\d+)", line)
                if m and int(m.group(1)) > 0:
                    per_class[ac]["last_candle"] = ts_str
                    per_class[ac]["connected"] = True
            # Even if no per-class candles, heartbeat proves bot is alive
            for ac in ["crypto", "forex", "stocks", "commodities"]:
                if "last_heartbeat" not in per_class[ac]:
                    per_class[ac]["last_heartbeat"] = ts_str
                per_class[ac]["last_heartbeat"] = ts_str

    # Market hours (UTC)
    hour = now.hour
    weekday = now.weekday()  # 0=Mon, 6=Sun
    market_open = {
        "crypto": True,  # 24/7
        "forex": 0 <= weekday <= 4 or (weekday == 6 and hour >= 22),  # Sun 22:00 - Fri 22:00
        "stocks": 0 <= weekday <= 4 and 13 <= hour < 20,  # Mon-Fri 13:30-20:00 UTC
        "commodities": 0 <= weekday <= 4 or (weekday == 6 and hour >= 23),  # ~23h/day
    }
    next_open = {
        "crypto": None,
        "forex": "Sun 22:00 UTC" if not market_open["forex"] else None,
        "stocks": "Mon-Fri 13:30 UTC" if not market_open["stocks"] else None,
        "commodities": "Sun 23:00 UTC" if not market_open["commodities"] else None,
    }

    # Override with heartbeat.json (real candle timestamps, written by bot on each candle)
    # Read AFTER log parsing so heartbeat.json takes precedence
    hb_data = None
    if heartbeat_path.exists():
        try:
            with open(heartbeat_path) as f:
                hb_data = json.load(f)
        except Exception:
            pass

    if hb_data and "per_class" in hb_data:
        for ac in ["crypto", "forex", "stocks", "commodities"]:
            hb_ac = hb_data["per_class"].get(ac, {})
            if hb_ac.get("last_candle_iso"):
                per_class[ac]["last_candle"] = hb_ac["last_candle_iso"]
                per_class[ac]["connected"] = True
                per_class[ac]["candles_total"] = hb_ac.get("candles_total", 0)
                per_class[ac]["symbols_active"] = hb_ac.get("symbols_active", 0)
                per_class[ac]["symbols_total"] = hb_ac.get("symbols_total", 0)

    # Check staleness (>10 min old) and compute age
    # Use the freshest of last_candle or last_heartbeat for staleness
    for ac, info in per_class.items():
        info["market_open"] = market_open.get(ac, True)
        info["next_open"] = next_open.get(ac)

        # Determine freshest timestamp (candle activity OR heartbeat)
        freshest_ts = info.get("last_candle")
        heartbeat_ts = info.get("last_heartbeat")
        if heartbeat_ts:
            if not freshest_ts:
                freshest_ts = heartbeat_ts
            else:
                try:
                    candle_dt = datetime.strptime(freshest_ts, "%Y-%m-%d %H:%M:%S")
                    hb_dt = datetime.strptime(heartbeat_ts, "%Y-%m-%d %H:%M:%S")
                    if hb_dt > candle_dt:
                        freshest_ts = heartbeat_ts
                except Exception:
                    pass

        if freshest_ts:
            try:
                last = datetime.strptime(freshest_ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                delta = (now - last).total_seconds()
                info["stale"] = delta > 600 and info["market_open"]  # Only stale if market is open
                info["age_seconds"] = int(delta)
                info["age_text"] = _format_age(int(delta))
                # Add heartbeat info
                if heartbeat_ts:
                    try:
                        hb_last = datetime.strptime(heartbeat_ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                        hb_age = int((now - hb_last).total_seconds())
                        info["heartbeat_age"] = _format_age(hb_age)
                        info["heartbeat_fresh"] = hb_age < 600
                    except Exception:
                        pass
            except Exception:
                info["stale"] = True
                info["age_text"] = "unknown"
        else:
            info["age_text"] = "no data yet"
            info["stale"] = False  # No data != stale (could be warming up)

    # Set exchange last_data and error age from per_class
    for exch_name, ac_key in [("Binance", "crypto"), ("OANDA", "forex"), ("Alpaca", "stocks")]:
        if per_class[ac_key]["last_candle"]:
            exchanges[exch_name]["last_data"] = per_class[ac_key]["last_candle"]
        exchanges[exch_name]["market_open"] = market_open.get(ac_key, True)
        exchanges[exch_name]["next_open"] = next_open.get(ac_key)
        # Error age: only show as "recent" if < 5 min old
        if exchanges[exch_name]["last_error"]:
            try:
                err_ts = datetime.strptime(exchanges[exch_name]["last_error"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                err_age = (now - err_ts).total_seconds()
                exchanges[exch_name]["error_recent"] = err_age < 300  # < 5 min
                exchanges[exch_name]["error_age_text"] = _format_age(int(err_age))
            except Exception:
                exchanges[exch_name]["error_recent"] = False
                exchanges[exch_name]["error_age_text"] = "unknown"
        else:
            exchanges[exch_name]["error_recent"] = False

    return jsonify({"exchanges": exchanges, "per_class": per_class})


@app.route("/api/public/trade/<trade_id>")
def api_trade_detail(trade_id: str):
    if not _journal_has_table("trades"):
        return jsonify({"trade_id": trade_id, "bars": [], "metadata": {}})
    rows = _query_journal("SELECT * FROM trades WHERE trade_id = ? LIMIT 1", (trade_id,))
    if not rows:
        return jsonify({"trade_id": trade_id, "bars": [], "metadata": {}})
    metadata = rows[0]
    bars: list[dict] = []
    if _journal_has_table("trade_bars"):
        bars = _query_journal(
            "SELECT * FROM trade_bars WHERE trade_id = ? ORDER BY bar_index ASC",
            (trade_id,),
        )
    return jsonify({"trade_id": trade_id, "bars": bars, "metadata": metadata})


# =====================================================================
#  New API Endpoints (active trades, candles, trade history)
# =====================================================================

COMMODITY_SYMBOLS = {"BCO_USD", "WTICO_USD", "XAG_USD", "XAU_USD"}
DATA_DIR = Path("data")


@app.route("/api/public/active-trades")
def api_active_trades():
    """Return all currently open positions across all bots."""
    bots = _discover_bots()
    now = datetime.now(timezone.utc)
    result: list[dict] = []
    for _tag, bot in bots.items():
        for trade in bot.get("active_trades", []):
            if not isinstance(trade, dict):
                continue
            # Trade dict uses internal keys: "entry" (not "entry_price"),
            # "rl_confidence" (not "confidence"), etc.
            entry_p = trade.get("entry") or trade.get("entry_price") or 0.0

            # Calculate hold time in hours from entry_time to now
            hold_time_hours = 0.0
            entry_time_str = trade.get("entry_time", "")
            if entry_time_str:
                try:
                    et = datetime.fromisoformat(entry_time_str)
                    if et.tzinfo is None:
                        et = et.replace(tzinfo=timezone.utc)
                    hold_time_hours = round((now - et).total_seconds() / 3600, 1)
                except Exception:
                    pass

            ac = bot.get("asset_class", "unknown")
            mult = DISPLAY_MULTIPLIERS.get(ac, 1.0)
            raw_pnl = float(trade.get("unrealized_pnl_usd") or 0.0)
            result.append({
                "symbol": trade.get("symbol", bot.get("symbol", _tag)),
                "direction": trade.get("direction", ""),
                "entry_price": float(entry_p),
                "sl": float(trade.get("sl", 0.0)),
                "tp": float(trade.get("tp", 0.0)),
                "entry_time": entry_time_str,
                "style": trade.get("style", ""),
                "tier": trade.get("tier") or trade.get("setup_tier", ""),
                "confidence": float(trade.get("rl_confidence") or trade.get("confidence") or 0.0),
                "asset_class": ac,
                "unrealized_pnl": raw_pnl * mult,
                "qty": float(trade.get("qty", 0.0)),
                "hold_time_hours": hold_time_hours,
                "candles_seen": int(trade.get("_candles_seen", 0)),
                "be_triggered": bool(trade.get("be_triggered", False)),
                "risk_pct": float(trade.get("risk_pct", 0.0)),
            })
    return jsonify(result)


@app.route("/api/public/candles/<path:symbol>")
def api_candles(symbol: str):
    """Return OHLCV candles for a symbol in Lightweight-Charts format."""
    try:
        limit = min(int(request.args.get("limit", 100)), 200)
    except (ValueError, TypeError):
        limit = 100

    # Priority 1: Live candle buffer from bot (live_results/candles/{symbol}.json)
    # These are written every 60s by the bot from its in-memory 5m buffer.
    sym_key = symbol.replace("/", "_").replace(":", "_")
    live_candle_path = RESULTS_DIR / "candles" / f"{sym_key}.json"
    if live_candle_path.exists():
        try:
            import json as _json
            with open(live_candle_path) as f:
                candles = _json.load(f)
            if candles:
                return jsonify(candles[-limit:])
        except Exception:
            pass

    # Priority 2: Historical data files (data/{class}/{symbol}_{tf}.parquet)
    tf = request.args.get("tf", "5m")
    subdir = _resolve_data_subdir(symbol)
    if subdir is None:
        return jsonify([])

    fname = f"{symbol}_{tf}.parquet"
    fpath = DATA_DIR / subdir / fname
    if not fpath.exists():
        return jsonify([])

    try:
        import pandas as pd
        df = pd.read_parquet(fpath)
        if df.empty:
            return jsonify([])
        df = df.tail(limit)
        candles_list: list[dict] = []
        for _, row in df.iterrows():
            ts = row.get("timestamp")
            if ts is None:
                continue
            if hasattr(ts, "timestamp"):
                unix_ts = int(ts.timestamp())
            else:
                unix_ts = int(ts)
            candles_list.append({
                "time": unix_ts,
                "open": float(row.get("open", 0)),
                "high": float(row.get("high", 0)),
                "low": float(row.get("low", 0)),
                "close": float(row.get("close", 0)),
                "volume": float(row.get("volume", 0)),
            })
        return jsonify(candles_list)
    except Exception:
        return jsonify([])


def _resolve_data_subdir(symbol: str) -> str | None:
    """Map a symbol name to its data subdirectory."""
    # Commodities (known set)
    if symbol in COMMODITY_SYMBOLS:
        return "commodities"
    # Crypto: contains USDT
    if "USDT" in symbol:
        return "crypto"
    # Forex: pattern like XXX_YYY where YYY is a currency code (not USDT)
    # Check forex dir first, then stocks
    if (DATA_DIR / "forex").exists():
        # Quick check: does a file with this symbol exist in forex?
        sample = DATA_DIR / "forex" / f"{symbol}_5m.parquet"
        if sample.exists():
            return "forex"
    if (DATA_DIR / "stocks").exists():
        sample = DATA_DIR / "stocks" / f"{symbol}_5m.parquet"
        if sample.exists():
            return "stocks"
    # Fallback: try all dirs
    for subdir in ("forex", "crypto", "stocks", "commodities"):
        sample = DATA_DIR / subdir / f"{symbol}_5m.parquet"
        if sample.exists():
            return subdir
    return None


@app.route("/api/public/trade-history")
def api_trade_history():
    """Return paginated trade history from journal DB."""
    try:
        limit = min(int(request.args.get("limit", 50)), 200)
    except (ValueError, TypeError):
        limit = 50
    try:
        offset = max(int(request.args.get("offset", 0)), 0)
    except (ValueError, TypeError):
        offset = 0
    asset_class = request.args.get("asset_class")

    if not _journal_has_table("trades"):
        return jsonify([])

    cols = _journal_columns()
    # Build SELECT with only columns that exist
    desired = [
        "trade_id", "symbol", "asset_class", "direction",
        "entry_time", "exit_time", "entry_price", "exit_price",
        "pnl_pct", "rr_actual", "risk_pct", "score", "xgb_confidence",
        "outcome", "exit_reason",
    ]
    select_cols = [c for c in desired if c in cols]
    if not select_cols:
        return jsonify([])

    select_str = ", ".join(select_cols)

    # Build WHERE clause
    where_parts: list[str] = []
    params: list = []
    # Only closed trades (have exit_time)
    if "exit_time" in cols:
        where_parts.append("exit_time IS NOT NULL")
    if asset_class and "asset_class" in cols:
        where_parts.append("asset_class = ?")
        params.append(asset_class)

    where_str = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""

    # ORDER BY exit_time DESC
    order_str = " ORDER BY exit_time DESC" if "exit_time" in cols else ""

    sql = f"SELECT {select_str} FROM trades{where_str}{order_str} LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    rows = _query_journal(sql, tuple(params))
    # Fill in missing keys with defaults so response shape is consistent
    result: list[dict] = []
    for row in rows:
        entry: dict = {}
        for col in desired:
            entry[col] = row.get(col)
        result.append(entry)
    return jsonify(result)


# =====================================================================
#  CORS
# =====================================================================

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


# =====================================================================
#  HTML Page
# =====================================================================

PUBLIC_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SMC Trading Bot -- Public Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://unpkg.com/lightweight-charts@4/dist/lightweight-charts.standalone.production.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#c9d1d9;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     font-size:14px;line-height:1.5}
a{color:#58a6ff;text-decoration:none}
a:hover{text-decoration:underline}

/* Header */
.header{position:fixed;top:0;left:0;right:0;z-index:100;
        background:#161b22;border-bottom:1px solid #30363d;
        padding:10px 20px;display:flex;align-items:center;gap:16px;flex-wrap:wrap}
.header h1{font-size:18px;color:#e6edf3;white-space:nowrap}
.status-dot{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:4px}
.status-dot.on{background:#3fb950;box-shadow:0 0 6px #3fb950}
.status-dot.off{background:#f85149;box-shadow:0 0 6px #f85149}
.header-info{font-size:13px;color:#8b949e;display:flex;gap:14px;flex-wrap:wrap;align-items:center}
.header-info span{white-space:nowrap}
.header-right{margin-left:auto;display:flex;gap:14px;align-items:center}
.session-badge{padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600;
               background:#1f2937;border:1px solid #30363d;color:#d2a8ff}

/* Layout */
.container{max-width:1400px;margin:0 auto;padding:70px 16px 24px}

/* Cards */
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:20px}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px}
.card-label{font-size:12px;color:#8b949e;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px}
.card-value{font-size:28px;font-weight:700;color:#e6edf3}
.card-sub{font-size:12px;color:#8b949e;margin-top:2px}

/* Colors */
.green{color:#3fb950}.red{color:#f85149}.yellow{color:#d29922}.blue{color:#58a6ff}
.bg-green{background:rgba(63,185,80,0.15)}.bg-red{background:rgba(248,81,73,0.15)}

/* Section */
.section{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin-bottom:16px}
.section-title{font-size:15px;font-weight:600;color:#e6edf3;margin-bottom:12px;
               padding-bottom:8px;border-bottom:1px solid #21262d}

/* Table */
.tbl{width:100%;border-collapse:collapse;font-size:13px}
.tbl th{text-align:left;padding:8px 10px;color:#8b949e;font-weight:600;
        border-bottom:1px solid #21262d;white-space:nowrap}
.tbl td{padding:6px 10px;border-bottom:1px solid #21262d1a}
.tbl tr:hover{background:#1c2129}
.tbl-scroll{overflow-x:auto;max-height:400px;overflow-y:auto}

/* Progress bar */
.progress-wrap{background:#21262d;border-radius:4px;height:16px;overflow:hidden;position:relative}
.progress-fill{height:100%;border-radius:4px;transition:width .5s}
.progress-text{position:absolute;top:0;left:0;right:0;height:16px;line-height:16px;
               text-align:center;font-size:11px;color:#e6edf3}

/* Log */
.log-box{background:#0d1117;border:1px solid #21262d;border-radius:6px;padding:8px;
         font-family:'SF Mono',Consolas,'Liberation Mono',Menlo,monospace;font-size:12px;
         max-height:300px;overflow-y:auto;line-height:1.6;overflow-wrap:break-word;word-break:break-all}
.log-box .info{color:#8b949e}.log-box .warn{color:#d29922}.log-box .error{color:#f85149}
.log-box .trade{color:#3fb950}

/* Near-miss */
.nm-list{max-height:320px;overflow-y:auto}
.nm-table{width:100%;border-collapse:collapse;font-size:0.82rem}
.nm-table th{text-align:left;padding:4px 6px;border-bottom:1px solid #444;color:#aaa;font-weight:500}
.nm-table td{padding:3px 6px;border-bottom:1px solid #333}
.nm-hot{background:rgba(76,175,80,0.15)}
.nm-warm{background:rgba(255,193,7,0.10)}
.nm-cold{}
.nm-neutral{opacity:0.5}
.nm-badge{padding:1px 6px;border-radius:3px;font-size:0.75rem;font-weight:600}
.nm-badge-alignment{background:#1565c0;color:#fff}
.nm-badge-xgb{background:#6a1b9a;color:#fff}
.nm-badge-neutral_bias{background:#555;color:#ccc}
.nm-badge-other{background:#333;color:#999}

/* Chart */
.chart-wrap{position:relative;height:280px}

/* Grid layouts */
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}

/* Empty state */
.empty{text-align:center;color:#484f58;padding:32px;font-size:14px}

/* Responsive */
@media(max-width:768px){
  .header{padding:8px 12px}
  .header h1{font-size:15px}
  .header-info{font-size:11px;gap:8px}
  .container{padding:62px 8px 16px}
  .cards{grid-template-columns:repeat(2,1fr);gap:8px}
  .card-value{font-size:22px}
  .grid-2,.grid-3{grid-template-columns:1fr}
  .chart-wrap{height:200px}
}

/* Active Trade Cards */
.trade-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px;margin-top:12px}
.trade-card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px;animation:fadeIn .3s ease}
.trade-card-header{display:flex;align-items:center;gap:10px;margin-bottom:8px;flex-wrap:wrap}
.trade-card-symbol{font-size:16px;font-weight:700;color:#e6edf3}
.trade-card-dir{padding:2px 8px;border-radius:4px;font-size:12px;font-weight:600;color:#fff}
.trade-card-dir.long{background:#238636}.trade-card-dir.short{background:#da3633}
.trade-card-tag{padding:2px 6px;border-radius:3px;font-size:11px;background:#21262d;color:#8b949e}
.trade-card-chart{width:100%;height:250px;border-radius:4px;overflow:hidden;margin-bottom:8px}
.trade-card-info{display:flex;gap:16px;font-size:12px;color:#8b949e;flex-wrap:wrap}
.trade-card-info span{white-space:nowrap}
.trade-card-info .label{color:#484f58}

/* Trade History Table */
.trade-history-table{width:100%;border-collapse:collapse;font-size:13px}
.trade-history-table th{text-align:left;padding:8px 10px;color:#8b949e;font-weight:600;
        border-bottom:1px solid #21262d;white-space:nowrap}
.trade-history-table td{padding:6px 10px;border-bottom:1px solid #21262d1a}
.trade-history-table tr:hover{background:#1c2129}
.win-row{background:rgba(63,185,80,0.08)}
.loss-row{background:rgba(248,81,73,0.08)}
.btn-load-more{display:inline-block;margin-top:12px;padding:8px 20px;border:1px solid #30363d;
               border-radius:6px;background:#21262d;color:#c9d1d9;cursor:pointer;font-size:13px}
.btn-load-more:hover{background:#30363d}

/* Fade in */
@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
@keyframes pulse{0%{opacity:1}50%{opacity:0.3}100%{opacity:1}}
.section,.card{animation:fadeIn .3s ease}
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <h1>SMC Trading Bot</h1>
  <div class="header-info">
    <span><span class="status-dot off" id="hd-dot"></span> <span id="hd-status">Checking...</span></span>
    <span id="hd-uptime"></span>
    <span id="hd-equity"></span>
  </div>
  <div class="header-right">
    <span class="session-badge" id="hd-session">--</span>
    <span style="font-size:11px;color:#484f58" id="hd-ts"></span>
  </div>
</div>

<div class="container">

  <!-- Stats Cards -->
  <div class="cards" id="stats-cards">
    <div class="card">
      <div class="card-label">Total Trades</div>
      <div class="card-value" id="c-trades">--</div>
      <div class="card-sub" id="c-active">0 open</div>
    </div>
    <div class="card">
      <div class="card-label">Win Rate</div>
      <div class="card-value" id="c-wr">--</div>
      <div class="card-sub" id="c-wins">0W / 0L</div>
    </div>
    <div class="card">
      <div class="card-label">Profit Factor</div>
      <div class="card-value" id="c-pf">--</div>
      <div class="card-sub" id="c-sharpe">Sharpe: --</div>
    </div>
    <div class="card">
      <div class="card-label">Total PnL</div>
      <div class="card-value" id="c-pnl">--</div>
      <div class="card-sub" id="c-pnl-dollar"></div>
      <div class="card-sub" id="c-dd">Max DD: --</div>
    </div>
    <div class="card">
      <div class="card-label">Unrealized PnL</div>
      <div class="card-value" id="c-upnl">--</div>
      <div class="card-sub" id="c-upnl-count">0 open positions</div>
      <div class="card-sub" id="c-upnl-breakdown" style="font-size:11px;margin-top:2px"></div>
    </div>
  </div>

  <!-- Period PnL Cards -->
  <div class="cards" id="period-cards" style="margin-top:8px">
    <div class="card">
      <div class="card-label">Daily PnL</div>
      <div class="card-value" id="c-daily-pnl">--</div>
      <div class="card-sub" id="c-daily-sub">0 trades</div>
    </div>
    <div class="card">
      <div class="card-label">Weekly PnL</div>
      <div class="card-value" id="c-weekly-pnl">--</div>
      <div class="card-sub" id="c-weekly-sub">0 trades</div>
    </div>
    <div class="card">
      <div class="card-label">Monthly PnL</div>
      <div class="card-value" id="c-monthly-pnl">--</div>
      <div class="card-sub" id="c-monthly-sub">0 trades</div>
    </div>
  </div>

  <!-- Active Trades with Charts -->
  <div class="section" id="active-trades-section">
    <div class="section-title">Active Trades <span id="at-count" style="font-size:12px;color:#8b949e;font-weight:400"></span></div>
    <div id="at-empty" class="empty">No active trades</div>
    <div class="trade-grid" id="at-grid"></div>
  </div>

  <!-- Trade History -->
  <div class="section">
    <div class="section-title">Trade History</div>
    <div class="tbl-scroll" style="max-height:560px;overflow-y:auto">
      <table class="trade-history-table" id="th-table">
        <thead style="position:sticky;top:0;background:#161b22;z-index:1"><tr>
          <th>Time</th><th>Symbol</th><th>Direction</th>
          <th>Entry</th><th>Exit</th><th>RR</th><th>Risk%</th><th>Conf</th>
          <th>PnL %</th><th>PnL $</th><th>Outcome</th><th>Exit Reason</th>
        </tr></thead>
        <tbody id="th-body"></tbody>
      </table>
    </div>
    <div id="th-empty" class="empty" style="display:none">No trade history yet</div>
    <button class="btn-load-more" id="th-more" style="display:none" onclick="loadMoreHistory()">Load More</button>
  </div>

  <!-- Equity Chart -->
  <div class="section">
    <div class="section-title">Equity Curve</div>
    <div class="chart-wrap"><canvas id="equity-chart"></canvas></div>
    <div class="empty" id="equity-empty" style="display:none">No equity data yet</div>
  </div>

  <div class="section">
    <div class="section-title">PnL Curve</div>
    <div class="chart-wrap"><canvas id="pnl-chart"></canvas></div>
    <div class="empty" id="pnl-empty" style="display:none">No PnL data yet</div>
  </div>

  <div class="grid-2">
    <!-- Asset Class Table -->
    <div class="section">
      <div class="section-title">Per Asset Class</div>
      <div class="tbl-scroll">
        <table class="tbl" id="class-table">
          <thead><tr><th>Class</th><th>Trades</th><th>WR</th><th>PnL</th><th>Open</th></tr></thead>
          <tbody id="class-body"><tr><td colspan="5" class="empty">Loading...</td></tr></tbody>
        </table>
      </div>
    </div>

    <!-- RL Analytics -->
    <div class="section">
      <div class="section-title">RL Analytics</div>
      <div id="rl-content"><div class="empty">Loading...</div></div>
    </div>
  </div>


  <div class="grid-2">
    <!-- Near-miss Monitor -->
    <div class="section">
      <div class="section-title">Near-Miss Monitor</div>
      <div class="nm-list" id="nm-list"><div class="empty">No near-misses recorded</div></div>
    </div>
    <div></div>
  </div>

  <div class="grid-2">
    <!-- Connection Status -->
    <div class="section">
      <div class="section-title">Connection Status</div>
      <table class="tbl" id="conn-table" style="border-spacing:0;">
        <thead><tr>
          <th style="padding:10px 16px;">Exchange</th>
          <th style="padding:10px 16px;">Status</th>
          <th style="padding:10px 16px;">Asset Class</th>
          <th style="padding:10px 16px;">Last Data</th>
          <th style="padding:10px 16px;">Last Error</th>
        </tr></thead>
        <tbody id="conn-body"><tr><td colspan="5" class="empty">Loading...</td></tr></tbody>
      </table>
    </div>

    <!-- Risk Dashboard -->
    <div class="section">
      <div class="section-title">Risk / Circuit Breakers</div>
      <div id="risk-content"><div class="empty">Loading...</div></div>
    </div>
  </div>

  <!-- Live Log Feed -->
  <div class="section">
    <div class="section-title">Live Log Feed</div>
    <div class="log-box" id="log-box"><span class="info">Waiting for logs...</span></div>
  </div>

  <!-- System Health -->
  <div class="section">
    <div class="section-title">System Health</div>
    <div class="grid-3" id="sys-bars">
      <div>
        <div class="card-label">CPU</div>
        <div class="progress-wrap"><div class="progress-fill" id="sys-cpu" style="width:0%;background:#3fb950"></div>
        <div class="progress-text" id="sys-cpu-t">--</div></div>
      </div>
      <div>
        <div class="card-label">RAM</div>
        <div class="progress-wrap"><div class="progress-fill" id="sys-ram" style="width:0%;background:#58a6ff"></div>
        <div class="progress-text" id="sys-ram-t">--</div></div>
      </div>
      <div>
        <div class="card-label">Disk</div>
        <div class="progress-wrap"><div class="progress-fill" id="sys-disk" style="width:0%;background:#d2a8ff"></div>
        <div class="progress-text" id="sys-disk-t">--</div></div>
      </div>
    </div>
    <div style="margin-top:8px;font-size:12px;color:#8b949e" id="sys-bot-mem">Bot RSS: --</div>
  </div>

  <div style="text-align:center;padding:16px;color:#484f58;font-size:12px">
    SMC Multi-Asset AAA++ Trading Bot &mdash; Public Read-Only Dashboard
  </div>
</div>

<script>
// ── Helpers ──────────────────────────────────────────────────────

function $(id){ return document.getElementById(id); }
function fmt(n, d){ return n != null ? Number(n).toFixed(d) : '--'; }
function pnlColor(v){ return v > 0 ? 'green' : v < 0 ? 'red' : ''; }
function escHtml(s){
  if(!s) return '';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── Status ───────────────────────────────────────────────────────

function updateStatus(d){
  var dot = $('hd-dot');
  dot.className = 'status-dot ' + (d.running ? 'on' : 'off');
  $('hd-status').textContent = d.running ? 'Running (PID '+d.pid+')' : 'Stopped';
  $('hd-uptime').textContent = d.uptime && d.uptime !== 'unknown' ? 'Up: '+d.uptime : '';
  $('hd-session').textContent = d.session || '--';
  $('hd-ts').textContent = d.timestamp || '';
}

// ── Overview ─────────────────────────────────────────────────────

function updateOverview(d){
  $('c-trades').textContent = d.total_trades || 0;
  $('c-active').textContent = (d.active_positions||0) + ' open';

  var wr = d.win_rate || 0;
  $('c-wr').textContent = fmt(wr,1) + '%';
  $('c-wr').className = 'card-value ' + (wr >= 50 ? 'green' : wr > 0 ? 'yellow' : '');

  var wins = d.total_wins || 0;
  var losses = (d.total_trades||0) - wins;
  $('c-wins').textContent = wins + 'W / ' + losses + 'L';

  var pf = d.profit_factor || 0;
  $('c-pf').textContent = pf > 100 ? 'INF' : fmt(pf,2);
  $('c-pf').className = 'card-value ' + (pf >= 1.5 ? 'green' : pf >= 1 ? 'yellow' : pf > 0 ? 'red' : '');
  $('c-sharpe').textContent = 'Sharpe: ' + fmt(d.sharpe,2);

  var pnl = d.total_pnl || 0;
  var pnlPct = d.total_pnl_pct || 0;
  $('c-pnl').textContent = (pnlPct >= 0 ? '+' : '') + fmt(pnlPct,2) + '%';
  $('c-pnl').className = 'card-value ' + pnlColor(pnl);
  $('c-pnl-dollar').innerHTML = '<span style="color:#8b949e;font-size:12px">' + (pnl >= 0 ? '+$' : '-$') + fmt(Math.abs(pnl),0) + '</span>';
  // Show worst per-broker DD (each account is a separate funded account)
  var ddLabel = 'Max DD: ' + fmt(d.max_dd_pct,2) + '%';
  var pb = d.per_broker || {};
  var worstBroker = '';
  var worstDD = 0;
  var brokerLabels = {'binance':'Crypto','oanda':'Forex','alpaca':'Stocks'};
  for(var bk in pb){ if(pb[bk].dd_pct > worstDD){ worstDD = pb[bk].dd_pct; worstBroker = bk; } }
  if(worstBroker) ddLabel += ' (' + (brokerLabels[worstBroker]||worstBroker) + ')';
  $('c-dd').textContent = ddLabel;
  $('c-dd').style.color = worstDD > 3 ? '#f85149' : worstDD > 1 ? '#d29922' : '#8b949e';

  $('hd-equity').textContent = d.total_equity ? ('Equity: $' + fmt(d.total_equity,2)) : '';
}

// ── Equity + PnL Charts (separate) ───────────────────────────────

var eqChart = null;
var pnlChart = null;

function updateEquity(data){
  var eqCanvas = $('equity-chart');
  var eqEmpty = $('equity-empty');
  var pnlCanvas = $('pnl-chart');
  var pnlEmpty = $('pnl-empty');

  if(!data || !data.length){
    if(eqCanvas) eqCanvas.style.display='none';
    if(eqEmpty) eqEmpty.style.display='block';
    if(pnlCanvas) pnlCanvas.style.display='none';
    if(pnlEmpty) pnlEmpty.style.display='block';
    return;
  }
  if(eqCanvas) eqCanvas.style.display='block';
  if(eqEmpty) eqEmpty.style.display='none';
  if(pnlCanvas) pnlCanvas.style.display='block';
  if(pnlEmpty) pnlEmpty.style.display='none';

  var labels = data.map(function(p){ return p.timestamp ? p.timestamp.substring(5,16) : ''; });
  var eqVals = data.map(function(p){ return p.equity; });
  var pnlVals = data.map(function(p){ return p.pnl; });

  // Tight equity Y-axis
  var eqMin = Math.min.apply(null, eqVals);
  var eqMax = Math.max.apply(null, eqVals);
  var eqPad = Math.max((eqMax - eqMin) * 0.3, eqMax * 0.005);

  // PnL color: green if positive, red if negative (current value)
  var lastPnl = pnlVals[pnlVals.length-1] || 0;
  var pnlColor = lastPnl >= 0 ? '#3fb950' : '#f85149';
  var pnlBg = lastPnl >= 0 ? 'rgba(63,185,80,0.1)' : 'rgba(248,81,73,0.1)';
  // Per-point colors for the line
  var pnlSegmentColors = pnlVals.map(function(v){return v>=0?'#3fb950':'#f85149'});
  var pnlPointColors = pnlVals.map(function(v){return v>=0?'#3fb950':'#f85149'});

  // ── Equity Chart ──
  if(eqChart){
    eqChart.data.labels = labels;
    eqChart.data.datasets[0].data = eqVals;
    eqChart.options.scales.y.suggestedMin = eqMin - eqPad;
    eqChart.options.scales.y.suggestedMax = eqMax + eqPad;
    eqChart.update('none');
  } else {
    eqChart = new Chart(eqCanvas, {
      type:'line',
      data:{labels:labels, datasets:[{
        label:'Portfolio Equity', data:eqVals,
        borderColor:'#58a6ff', backgroundColor:'rgba(88,166,255,0.08)',
        fill:true, tension:0.3, pointRadius:3, pointBackgroundColor:'#58a6ff', borderWidth:2
      }]},
      options:{
        responsive:true, maintainAspectRatio:false, animation:false,
        plugins:{legend:{labels:{color:'#8b949e',font:{size:11}}}},
        scales:{
          x:{ticks:{color:'#484f58',maxTicksLimit:12,font:{size:10}},grid:{color:'#21262d'}},
          y:{suggestedMin:eqMin-eqPad,suggestedMax:eqMax+eqPad,ticks:{color:'#58a6ff',font:{size:10},maxTicksLimit:6,callback:function(v){return '$'+Math.round(v).toLocaleString()}},grid:{color:'#21262d'}}
        }
      }
    });
  }

  // ── PnL Chart (2 datasets: green above 0, red below 0) ──
  if(pnlChart){
    pnlChart.data.labels = labels;
    pnlChart.data.datasets[0].data = pnlVals;
    pnlChart.data.datasets[1].data = pnlVals;
    pnlChart.update('none');
  } else {
    pnlChart = new Chart(pnlCanvas, {
      type:'line',
      data:{labels:labels, datasets:[
        {label:'PnL', data:pnlVals,
         borderColor:'#3fb950', backgroundColor:'rgba(63,185,80,0.12)',
         fill:{target:'origin',above:'rgba(63,185,80,0.12)',below:'transparent'},
         segment:{borderColor:function(ctx){return ctx.p0.parsed.y>=0&&ctx.p1.parsed.y>=0?'#3fb950':'#f85149'}},
         tension:0.3, pointRadius:5, pointBorderWidth:2,
         pointBackgroundColor:pnlPointColors, pointBorderColor:pnlPointColors, borderWidth:2},
        {label:'_neg', data:pnlVals,
         borderColor:'transparent', backgroundColor:'rgba(248,81,73,0.12)',
         fill:{target:'origin',above:'transparent',below:'rgba(248,81,73,0.12)'},
         tension:0.3, pointRadius:0, borderWidth:0}
      ]},
      options:{
        responsive:true, maintainAspectRatio:false, animation:false,
        plugins:{legend:{labels:{filter:function(item){return item.text!=='_neg'},color:'#8b949e',font:{size:11}}}},
        scales:{
          x:{ticks:{color:'#484f58',maxTicksLimit:12,font:{size:10}},grid:{color:'#21262d'}},
          y:{ticks:{color:function(ctx){return ctx.tick.value>=0?'#3fb950':'#f85149'},font:{size:10},callback:function(v){return (v>=0?'+$':'-$')+Math.abs(v).toLocaleString()}},grid:{color:'#21262d'}}
        }
      }
    });
  }
}

// ── Per-Class Table ──────────────────────────────────────────────

function updatePerClass(d){
  var body = $('class-body');
  var classes = ['crypto','forex','stocks','commodities'];
  var html = '';
  var anyData = false;
  for(var i=0; i<classes.length; i++){
    var ac = classes[i];
    var info = d[ac];
    if(!info) info = {trades:0, wins:0, pnl:0, win_rate:0, active:0};
    if(info.trades > 0) anyData = true;
    html += '<tr><td style="text-transform:capitalize;font-weight:600">'+ac+'</td>';
    html += '<td>'+info.trades+'</td>';
    html += '<td>'+fmt(info.win_rate,1)+'%</td>';
    html += '<td class="'+pnlColor(info.pnl)+'">'+fmt(info.pnl,4)+'</td>';
    html += '<td>'+( info.active||0)+'</td></tr>';
  }
  // Include unknown class if present
  for(var k in d){
    if(classes.indexOf(k) === -1 && d[k].trades > 0){
      var u = d[k];
      html += '<tr><td>'+escHtml(k)+'</td><td>'+u.trades+'</td><td>'+fmt(u.win_rate,1)+'%</td>';
      html += '<td class="'+pnlColor(u.pnl)+'">'+fmt(u.pnl,4)+'</td><td>'+(u.active||0)+'</td></tr>';
    }
  }
  if(!anyData && !html) html = '<tr><td colspan="5" class="empty">No data yet</td></tr>';
  body.innerHTML = html;
}

// ── RL Analytics ─────────────────────────────────────────────────

function updateRL(d){
  var el = $('rl-content');
  var ef = d.entry_filter || {};
  var total = (ef.accepted||0)+(ef.rejected||0);
  var html = '<table class="tbl">';
  html += '<tr><td>Entry Filter Accepted</td><td class="green">'+(ef.accepted||0)+'</td></tr>';
  html += '<tr><td>Entry Filter Rejected</td><td class="red">'+(ef.rejected||0)+'</td></tr>';
  html += '<tr><td>Acceptance Rate</td><td class="blue">'+fmt(ef.rate,1)+'%</td></tr>';
  html += '<tr><td>TP Adjustments</td><td>'+(d.tp_adjusted||0)+'</td></tr>';
  html += '<tr><td>BE Triggered</td><td>'+(d.be_triggered||0)+'</td></tr>';
  html += '<tr><td>Models Loaded</td><td>'+(d.models_loaded?d.models_loaded.length:0)+'</td></tr>';
  html += '</table>';
  if(total === 0) html += '<div class="empty" style="margin-top:8px">No RL decisions recorded yet</div>';
  el.innerHTML = html;
}


// ── Near-misses ──────────────────────────────────────────────────

function updateNearMisses(data){
  var el = $('nm-list');
  if(!data || !data.length){
    el.innerHTML = '<div class="empty">No near-misses recorded</div>';
    return;
  }
  var html = '<table class="nm-table"><thead><tr>' +
    '<th>Time</th><th>Symbol</th><th>Type</th><th>Score</th><th>Thresh</th><th>Gap</th><th>Dir</th><th>Class</th>' +
    '</tr></thead><tbody>';
  data.slice().reverse().forEach(function(nm){
    var gap = nm.gap || 0;
    var cls = '';
    if(nm.type === 'OTHER' || nm.type === 'NEUTRAL_BIAS') cls = 'nm-neutral';
    else if(gap <= 0.03) cls = 'nm-hot';
    else if(gap <= 0.08) cls = 'nm-warm';
    else cls = 'nm-cold';
    var timeShort = (nm.time || '').substring(11, 19);
    html += '<tr class="' + cls + '">' +
      '<td>' + escHtml(timeShort) + '</td>' +
      '<td>' + escHtml(nm.symbol || '') + '</td>' +
      '<td><span class="nm-badge nm-badge-' + escHtml((nm.type||'other').toLowerCase()) + '">' + escHtml(nm.type || '?') + '</span></td>' +
      '<td>' + (nm.score ? nm.score.toFixed(3) : '-') + '</td>' +
      '<td>' + (nm.threshold ? nm.threshold.toFixed(2) : '-') + '</td>' +
      '<td>' + (gap ? gap.toFixed(3) : '-') + '</td>' +
      '<td>' + escHtml(nm.direction || '-') + '</td>' +
      '<td>' + escHtml(nm.asset_class || '-') + '</td>' +
      '</tr>';
  });
  html += '</tbody></table>';
  el.innerHTML = html;
}

// ── Risk Dashboard ───────────────────────────────────────────────

function renderProgressBar(label, current, limit, color){
  var pct = limit > 0 ? Math.min(Math.abs(current)/limit*100, 100) : 0;
  var barColor = pct >= 80 ? '#f85149' : pct >= 50 ? '#d29922' : color || '#3fb950';
  return '<div style="margin-bottom:10px"><div class="card-label">'+label+
    ' ('+fmt(Math.abs(current),2)+'% / '+limit+'%)</div>'+
    '<div class="progress-wrap"><div class="progress-fill" style="width:'+pct+'%;background:'+barColor+'"></div>'+
    '<div class="progress-text">'+fmt(pct,0)+'%</div></div></div>';
}

function updateRisk(d){
  var el = $('risk-content');
  var pb = d.per_broker || {};
  var brokerNames = {'binance':'Binance (Crypto)','oanda':'OANDA (Forex+Commod.)','alpaca':'Alpaca (Stocks)'};
  var brokerOrder = ['binance','oanda','alpaca'];
  var html = '';

  for(var bi=0; bi<brokerOrder.length; bi++){
    var bk = brokerOrder[bi];
    var b = pb[bk];
    if(!b) continue;
    var bName = brokerNames[bk] || bk;
    var statusCol = b.status === 'CLEAR' ? '#3fb950' : '#f85149';

    html += '<div style="background:#161b22;border:1px solid #21262d;border-radius:8px;padding:10px 12px;margin-bottom:8px">';
    html += '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">';
    html += '<span style="font-size:13px;font-weight:600;color:#e6edf3">'+escHtml(bName)+'</span>';
    html += '<span style="font-size:11px;font-weight:600;color:'+statusCol+'">'+escHtml(b.status||'CLEAR')+'</span>';
    html += '</div>';
    html += renderProgressBar('Daily DD', Math.abs(b.daily_pnl_pct||0), d.daily_loss_limit||3, '#3fb950');
    html += renderProgressBar('Weekly DD', Math.abs(b.weekly_pnl_pct||0), d.weekly_loss_limit||5, '#58a6ff');
    html += renderProgressBar('All-Time DD', b.dd_pct||0, d.alltime_dd_limit||8, '#d2a8ff');
    html += renderProgressBar('Open Risk', b.heat_pct||0, 6, '#d29922');
    html += '<div style="font-size:11px;color:#8b949e;margin-top:4px">';
    html += 'PnL: <span style="color:'+(b.pnl>=0?'#3fb950':'#f85149')+'">'+(b.pnl>=0?'+$':'-$')+fmt(Math.abs(b.pnl),0)+' ('+fmt(b.pnl_pct,2)+'%)</span>';
    html += ' · Open: '+(b.active||0)+' · Budget: '+fmt(b.risk_budget||0,2)+'%';
    html += '</div>';

    // Breaker alerts for this broker
    var bkBreakers = [];
    if(b.daily_breaker) bkBreakers.push('DAILY STOP');
    if(b.weekly_breaker) bkBreakers.push('WEEKLY PAUSE');
    if(b.alltime_breaker) bkBreakers.push('ALL-TIME STOP');
    if(b.heat_breaker) bkBreakers.push('HEAT LIMIT');
    if(bkBreakers.length){
      html += '<div style="font-size:11px;font-weight:600;color:#f85149;margin-top:3px">⚠ '+bkBreakers.join(' | ')+'</div>';
    }
    html += '</div>';
  }

  if(!html) html = '<div class="empty">No broker data</div>';
  el.innerHTML = html;
}

// ── Log Feed ─────────────────────────────────────────────────────

function updateLogs(lines){
  var box = $('log-box');
  if(!lines || !lines.length){
    box.innerHTML = '<span class="info">No logs available</span>';
    return;
  }
  var html = '';
  for(var i=0; i<lines.length; i++){
    var line = escHtml(lines[i]);
    var cls = 'info';
    if(line.indexOf('[ERROR]') !== -1) cls = 'error';
    else if(line.indexOf('[WARN') !== -1) cls = 'warn';
    else if(line.indexOf('OPEN [') !== -1 || line.indexOf('CLOSE ') !== -1) cls = 'trade';
    html += '<div class="'+cls+'">'+line+'</div>';
  }
  box.innerHTML = html;
  // Auto-scroll to bottom
  box.scrollTop = box.scrollHeight;
}

// ── System Health ────────────────────────────────────────────────

function updateSystem(d){
  function setBar(id, pct, txt){
    var fill = $(id);
    var label = $(id+'-t');
    if(fill) fill.style.width = Math.min(pct,100)+'%';
    if(label) label.textContent = txt;
    // Color code
    if(fill){
      if(pct >= 90) fill.style.background = '#f85149';
      else if(pct >= 70) fill.style.background = '#d29922';
    }
  }
  setBar('sys-cpu', d.cpu_pct||0, fmt(d.cpu_pct,1)+'%');
  setBar('sys-ram', d.ram_pct||0, (d.ram_used_mb||0)+'MB / '+(d.ram_total_mb||0)+'MB');
  setBar('sys-disk', d.disk_pct||0, (d.disk_used_gb||0)+'GB / '+(d.disk_total_gb||0)+'GB');
  $('sys-bot-mem').textContent = 'Bot RSS: ' + (d.bot_rss_mb ? fmt(d.bot_rss_mb,1)+' MB' : 'N/A');
}

// ── Fetch helpers ────────────────────────────────────────────────

function fetchJ(url, cb, retries){
  retries = retries || 0;
  fetch(url).then(function(r){
    if(!r.ok) throw new Error('HTTP ' + r.status);
    return r.json();
  }).then(cb).catch(function(e){
    console.warn('fetchJ fail: ' + url + ' (' + e.message + ')');
    if(retries < 2) setTimeout(function(){ fetchJ(url, cb, retries+1); }, 2000);
  });
}

// ── Period Stats (Daily / Weekly / Monthly) ─────────────────────

function updatePeriodStats(d){
  function renderPeriod(prefix, p){
    if(!p || !p.trades){
      $(prefix+'-pnl').textContent = '--';
      $(prefix+'-pnl').className = 'card-value';
      $(prefix+'-sub').textContent = '0 trades';
      return;
    }
    var pct = p.pnl_pct || 0;
    var dollar = p.pnl_dollar || 0;
    $(prefix+'-pnl').textContent = (pct >= 0 ? '+' : '') + fmt(pct,2) + '%';
    $(prefix+'-pnl').className = 'card-value ' + pnlColor(pct);
    var wr = p.trades > 0 ? (p.wins/p.trades*100) : 0;
    var sub = (dollar >= 0 ? '+$' : '-$') + fmt(Math.abs(dollar),0);
    sub += ' · ' + p.trades + ' trades · ' + fmt(wr,0) + '% WR';
    if(p.dd_pct > 0) sub += ' · DD: ' + fmt(p.dd_pct,2) + '%';
    $(prefix+'-sub').innerHTML = '<span style="color:#8b949e">' + sub + '</span>';
  }
  renderPeriod('c-daily', d.daily);
  renderPeriod('c-weekly', d.weekly);
  renderPeriod('c-monthly', d.monthly);
}

// ── Initial load + polling ───────────────────────────────────────

function loadAll(){
  fetchJ('/api/public/status', updateStatus);
  fetchJ('/api/public/overview', updateOverview);
  fetchJ('/api/public/equity', updateEquity);
  fetchJ('/api/public/per-class', updatePerClass);
  fetchJ('/api/public/period-stats', updatePeriodStats);
  fetchJ('/api/public/rl-stats', updateRL);
  fetchJ('/api/public/near-misses', updateNearMisses);
  fetchJ('/api/public/circuit-breaker', updateRisk);
  fetchJ('/api/public/logs', updateLogs);
  fetchJ('/api/public/system', updateSystem);
}

// Initial load
loadAll();

// Polling at different intervals
setInterval(function(){ fetchJ('/api/public/status', updateStatus); }, 10000);
setInterval(function(){ fetchJ('/api/public/overview', updateOverview); }, 30000);
setInterval(function(){ fetchJ('/api/public/equity', updateEquity); }, 60000);
setInterval(function(){ fetchJ('/api/public/per-class', updatePerClass); }, 30000);
setInterval(function(){ fetchJ('/api/public/period-stats', updatePeriodStats); }, 30000);
setInterval(function(){ fetchJ('/api/public/rl-stats', updateRL); }, 60000);
setInterval(function(){ fetchJ('/api/public/near-misses', updateNearMisses); }, 30000);
setInterval(function(){ fetchJ('/api/public/circuit-breaker', updateRisk); }, 15000);
setInterval(function(){ fetchJ('/api/public/logs', updateLogs); }, 5000);
setInterval(function(){ fetchJ('/api/public/system', updateSystem); }, 30000);

// ── Connection Status ────────────────────────────────────────────
function updateConnections(d){
  var b = document.getElementById('conn-body');
  if(!b) return;
  var rows = '';
  var exMap = {'Binance':'crypto','OANDA':'forex','Alpaca':'stocks'};
  var allClasses = [
    {name:'Binance', ac:'crypto'},
    {name:'OANDA', ac:'forex'},
    {name:'Alpaca', ac:'stocks'},
    {name:'OANDA', ac:'commodities'},
  ];
  for(var i=0; i<allClasses.length; i++){
    var item = allClasses[i];
    var ex = d.exchanges[item.name] || {};
    var pc = d.per_class[item.ac] || {};
    var isOpen = pc.market_open !== false;
    var isConnected = ex.status === 'connected' || pc.connected;

    // Big status indicator
    var dotColor, dotStyle='', statusText;
    if(!isOpen){
      dotColor = '#666'; statusText = 'Market Closed';
    } else if(pc.stale){
      dotColor = '#f85149'; dotStyle = 'animation:pulse 1s infinite;'; statusText = 'STALE';
    } else if(isConnected){
      dotColor = '#3fb950'; statusText = 'Healthy';
    } else {
      dotColor = '#d29922'; statusText = 'Connecting...';
    }
    var dot = '<span style="display:inline-block;width:14px;height:14px;border-radius:50%;background:'+dotColor+';margin-right:8px;vertical-align:middle;'+dotStyle+'"></span>';

    // Market badge
    var badge = '';
    if(!isOpen){
      var nextOpen = pc.next_open || ex.next_open || '';
      badge = '<span style="background:#333;color:#999;padding:2px 8px;border-radius:4px;font-size:12px;margin-left:6px;">Closed'+(nextOpen ? ' — opens '+nextOpen : '')+'</span>';
    }

    // Age text + heartbeat info
    var ageText = pc.age_text || 'no data';
    var ageColor = '#c9d1d9';
    if(pc.stale && isOpen){ ageColor='#f85149'; ageText='<b>'+ageText+' STALE</b>'; }
    else if(pc.age_seconds > 300 && isOpen){ ageColor='#d29922'; }
    else if(pc.age_seconds !== undefined && pc.age_seconds <= 300){ ageColor='#3fb950'; }
    if(pc.heartbeat_age){ ageText += ' <span style="color:#8b949e;font-size:11px;">(heartbeat: '+pc.heartbeat_age+')</span>'; }
    if(pc.symbols_active !== undefined){ ageText += ' <span style="color:#8b949e;font-size:11px;">['+pc.symbols_active+'/'+pc.symbols_total+' symbols]</span>'; }

    // Error display: red only if recent (<5 min), gray if old
    var errHtml;
    if(!ex.last_error || item.ac==='commodities'){
      errHtml = '<span style="color:#3fb950;">none</span>';
    } else if(ex.error_recent){
      errHtml = '<span style="color:#f85149;">'+ex.error_age_text+'</span>';
    } else {
      errHtml = '<span style="color:#666;">'+ex.error_age_text+' (resolved)</span>';
    }

    rows += '<tr>'
      + '<td style="padding:12px 16px;"><b>'+item.name+'</b></td>'
      + '<td style="padding:12px 16px;">'+dot+statusText+badge+'</td>'
      + '<td style="padding:12px 16px;">'+item.ac+'</td>'
      + '<td style="padding:12px 16px;color:'+ageColor+'">'+ageText+'</td>'
      + '<td style="padding:12px 16px;">'+errHtml+'</td>'
      + '</tr>';
  }
  b.innerHTML = rows;
}
fetchJ('/api/public/connections', updateConnections);
setInterval(function(){ fetchJ('/api/public/connections', updateConnections); }, 15000);

// ── Active Trades with Charts + Unrealized PnL ──────────────────
var _atCharts = {};
var _atSeries = {};  // reserved for future incremental candle updates
var _atSymSet = '';  // serialised symbol+direction key for change detection

function _autoPriceFormat(price){
  if(price < 0.001) return { type:'price', precision:8, minMove:0.00000001 };
  if(price < 0.1)   return { type:'price', precision:6, minMove:0.000001 };
  if(price < 1)     return { type:'price', precision:5, minMove:0.00001 };
  if(price < 100)   return { type:'price', precision:4, minMove:0.0001 };
  if(price < 10000) return { type:'price', precision:2, minMove:0.01 };
  return { type:'price', precision:0, minMove:1 };
}

function _buildSymKey(trades){
  if(!trades || !trades.length) return '';
  var parts = [];
  for(var i=0;i<trades.length;i++){
    parts.push((trades[i].symbol||'')+'|'+(trades[i].direction||''));
  }
  return parts.sort().join(',');
}

function updateActiveTrades(trades){
  var grid = $('at-grid');
  var empty = $('at-empty');
  var countEl = $('at-count');
  var upnlEl = $('c-upnl');
  var upnlCountEl = $('c-upnl-count');
  var upnlBreakdown = $('c-upnl-breakdown');
  if(!grid) return;

  // Compute combined unrealized PnL + per-class breakdown
  var totalUpnl = 0;
  var posCount = trades ? trades.length : 0;
  var classPnl = {};
  if(trades && trades.length > 0){
    for(var i=0;i<trades.length;i++){
      var upnl = trades[i].unrealized_pnl || 0;
      totalUpnl += upnl;
      var ac = trades[i].asset_class || 'unknown';
      classPnl[ac] = (classPnl[ac] || 0) + upnl;
    }
  }
  if(upnlEl){
    upnlEl.textContent = (totalUpnl >= 0 ? '+$' : '-$') + fmt(Math.abs(totalUpnl), 2);
    upnlEl.className = 'card-value ' + pnlColor(totalUpnl);
  }
  if(upnlCountEl) upnlCountEl.textContent = posCount + ' open position' + (posCount !== 1 ? 's' : '');
  // Per-class breakdown
  if(upnlBreakdown){
    var bk = [];
    for(var cls in classPnl){
      var v = classPnl[cls];
      var prefix = cls.charAt(0).toUpperCase()+cls.slice(1)+': ';
      bk.push(prefix + (v >= 0 ? '+$' : '-$') + fmt(Math.abs(v), 2));
    }
    upnlBreakdown.innerHTML = bk.length > 0 ? bk.join(' &nbsp;|&nbsp; ') : '';
  }

  if(!trades || trades.length === 0){
    empty.style.display = '';
    grid.innerHTML = '';
    if(countEl) countEl.textContent = '';
    for(var k in _atCharts){ try{ _atCharts[k].remove(); }catch(e){} }
    _atCharts = {};
    _atSeries = {};
    _atSymSet = '';
    return;
  }
  empty.style.display = 'none';
  if(countEl) countEl.textContent = '(' + trades.length + ')';

  // Check if trade set changed — if not, just update PnL values in-place
  var newSymSet = _buildSymKey(trades);
  if(newSymSet === _atSymSet && _atSymSet !== ''){
    // Update PnL values in existing cards without rebuilding
    for(var i=0;i<trades.length;i++){
      var t = trades[i];
      var symId = (t.symbol||'').replace(/[^a-zA-Z0-9]/g, '_');
      var pnlSpan = document.getElementById('pnl-' + symId);
      if(pnlSpan){
        var pnl = t.unrealized_pnl || 0;
        pnlSpan.className = pnlColor(pnl);
        pnlSpan.textContent = (pnl>=0?'+$':'-$') + fmt(Math.abs(pnl),2);
      }
    }
    return;
  }
  _atSymSet = newSymSet;

  // Destroy old charts before full rebuild
  for(var k in _atCharts){ try{ _atCharts[k].remove(); }catch(e){} }
  _atCharts = {};
  _atSeries = {};

  // Build cards
  var html = '';
  for(var i=0;i<trades.length;i++){
    var t = trades[i];
    var sym = t.symbol || '--';
    var symId = sym.replace(/[^a-zA-Z0-9]/g, '_');
    var dir = t.direction || 'long';
    var dirCls = dir === 'short' ? 'short' : 'long';
    var pnl = t.unrealized_pnl || 0;
    html += '<div class="trade-card">';
    html += '<div class="trade-card-header">';
    html += '<span class="trade-card-symbol">' + escHtml(sym) + '</span>';
    html += '<span class="trade-card-dir ' + dirCls + '">' + dir.toUpperCase() + '</span>';
    if(t.style) html += '<span class="trade-card-tag">' + escHtml(t.style) + '</span>';
    if(t.confidence) html += '<span class="trade-card-tag">conf: ' + fmt(t.confidence,3) + '</span>';
    html += '</div>';
    html += '<div class="trade-card-chart" id="chart-' + symId + '"></div>';
    html += '<div class="trade-card-info">';
    html += '<span><span class="label">Entry:</span> ' + fmt(t.entry_price,5) + '</span>';
    html += '<span><span class="label">SL:</span> <span class="red">' + fmt(t.sl,5) + '</span></span>';
    html += '<span><span class="label">TP:</span> <span class="green">' + fmt(t.tp,5) + '</span></span>';
    html += '<span><span class="label">PnL:</span> <span id="pnl-' + symId + '" class="' + pnlColor(pnl) + '">' + (pnl>=0?'+$':'-$') + fmt(Math.abs(pnl),2) + '</span></span>';
    var hh = t.hold_time_hours || 0;
    var holdStr = hh < 1 ? Math.round(hh*60)+'m' : (hh < 24 ? fmt(hh,1)+'h' : fmt(hh/24,1)+'d');
    html += '<span><span class="label">Hold:</span> ' + holdStr + '</span>';
    html += '</div></div>';
  }
  grid.innerHTML = html;

  // Create charts for each trade
  for(var i=0;i<trades.length;i++){
    (function(trade){
      var sym = trade.symbol || '';
      var symId = sym.replace(/[^a-zA-Z0-9]/g, '_');
      var container = document.getElementById('chart-' + symId);
      if(!container || !window.LightweightCharts) return;

      var chart = LightweightCharts.createChart(container, {
        width: container.clientWidth,
        height: 250,
        layout: { background: { color: '#0d1117' }, textColor: '#8b949e' },
        grid: { vertLines: { color: '#21262d' }, horzLines: { color: '#21262d' } },
        crosshair: { mode: 0 },
        timeScale: { timeVisible: true, secondsVisible: false, borderColor: '#30363d' },
        rightPriceScale: { borderColor: '#30363d' },
      });
      _atCharts[symId] = chart;

      var series = chart.addCandlestickSeries({
        upColor: '#3fb950', downColor: '#f85149',
        borderUpColor: '#3fb950', borderDownColor: '#f85149',
        wickUpColor: '#3fb950', wickDownColor: '#f85149',
      });
      _atSeries[symId] = series;

      // Fetch candles — sanitize URL to match bot export (replace / and : with _)
      var candleUrl = '/api/public/candles/' + sym.replace(/\//g, '_').replace(/:/g, '_') + '?tf=5m&limit=100';
      fetchJ(candleUrl, function(candles){
        if(!candles || !candles.length){
          console.warn('No candle data for ' + sym);
          return;
        }

        // Auto-detect price format from first candle
        var priceFmt = _autoPriceFormat(candles[0].close || candles[0].open || 1);
        series.applyOptions({ priceFormat: priceFmt });

        series.setData(candles);

        // Price lines
        if(trade.entry_price){
          series.createPriceLine({
            price: trade.entry_price, color: '#58a6ff',
            lineWidth: 2, lineStyle: 0, axisLabelVisible: true, title: 'Entry'
          });
        }
        if(trade.sl){
          series.createPriceLine({
            price: trade.sl, color: '#f85149',
            lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: 'SL'
          });
        }
        if(trade.tp){
          series.createPriceLine({
            price: trade.tp, color: '#3fb950',
            lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: 'TP'
          });
        }
        chart.timeScale().fitContent();
      });
    })(trades[i]);
  }
}

fetchJ('/api/public/active-trades', updateActiveTrades);
setInterval(function(){ fetchJ('/api/public/active-trades', updateActiveTrades); }, 30000);

// ── Trade History ────────────────────────────────────────────────
var _thOffset = 0;
var _thLimit = 50;

function renderTradeHistory(trades, append){
  var tbody = $('th-body');
  var empty = $('th-empty');
  var btn = $('th-more');
  if(!tbody) return;

  if(!trades || trades.length === 0){
    if(!append){
      tbody.innerHTML = '';
      if(empty) empty.style.display = '';
      if(btn) btn.style.display = 'none';
    }
    if(btn) btn.style.display = 'none';
    return;
  }
  if(empty) empty.style.display = 'none';

  var html = append ? '' : '';
  for(var i=0;i<trades.length;i++){
    var t = trades[i];
    var isWin = t.outcome === 'win';
    var rowCls = isWin ? 'win-row' : (t.outcome === 'loss' ? 'loss-row' : '');
    var pnlPct = (t.pnl_pct || 0) * 100;
    var pnlDollar = (t.pnl_pct || 0) * 100000;
    var dir = t.direction ? t.direction.charAt(0).toUpperCase() + t.direction.slice(1) : '--';
    var exitTime = t.exit_time ? t.exit_time.substring(0,16) : '--';
    html += '<tr class="'+rowCls+'">';
    html += '<td style="white-space:nowrap">'+escHtml(exitTime)+'</td>';
    html += '<td>'+escHtml(t.symbol||'--')+'</td>';
    html += '<td><span class="'+(t.direction==='long'?'green':'red')+'">'+dir+'</span></td>';
    html += '<td>'+fmt(t.entry_price,5)+'</td>';
    html += '<td>'+fmt(t.exit_price,5)+'</td>';
    html += '<td>'+fmt(t.rr_actual,2)+'</td>';
    html += '<td>'+((t.risk_pct||0)*100).toFixed(2)+'%</td>';
    html += '<td>'+fmt(t.xgb_confidence||t.score,3)+'</td>';
    html += '<td class="'+pnlColor(pnlPct)+'">'+(pnlPct>=0?'+':'')+pnlPct.toFixed(2)+'%</td>';
    html += '<td class="'+pnlColor(pnlDollar)+'">'+(pnlDollar>=0?'+$':'-$')+Math.abs(pnlDollar).toFixed(2)+'</td>';
    html += '<td><span class="'+(isWin?'green':'red')+'">'+(t.outcome||'--')+'</span></td>';
    html += '<td>'+escHtml(t.exit_reason||'--')+'</td>';
    html += '</tr>';
  }
  if(append) tbody.innerHTML += html;
  else tbody.innerHTML = html;

  if(btn) btn.style.display = trades.length >= _thLimit ? '' : 'none';
}

function loadMoreHistory(){
  _thOffset += _thLimit;
  fetchJ('/api/public/trade-history?limit='+_thLimit+'&offset='+_thOffset, function(d){ renderTradeHistory(d, true); });
}

fetchJ('/api/public/trade-history?limit='+_thLimit+'&offset=0', function(d){ renderTradeHistory(d, false); });
setInterval(function(){ _thOffset=0; fetchJ('/api/public/trade-history?limit='+_thLimit+'&offset=0', function(d){ renderTradeHistory(d, false); }); }, 60000);
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PUBLIC_HTML)


# =====================================================================
#  Main
# =====================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Public Trading Dashboard")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    print(f"Public dashboard starting on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
