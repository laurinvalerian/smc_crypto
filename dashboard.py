"""
Paper Trading Dashboard — lightweight Flask app with auto-refresh.
Reads from live_results/ (bot state JSONs, equity CSVs, log files).
Includes kill switch, pause/resume, and component toggle control panel.

Usage: python3 dashboard.py [--port 8080]
"""
from __future__ import annotations

import csv
import json
import os
import re
import signal as _signal
import sqlite3
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import secrets
from functools import wraps

import yaml
from flask import Flask, jsonify, render_template_string, request, session, redirect, url_for

app = Flask(__name__)
RESULTS_DIR = Path("live_results")

# ── Config ─────────────────────────────────────────────────────────
try:
    with open("config/default_config.yaml") as _f:
        _cfg = yaml.safe_load(_f)
    DASHBOARD_PIN = str(_cfg.get("dashboard", {}).get("pin", "1234"))
    DASHBOARD_USER = str(_cfg.get("dashboard", {}).get("username", "admin"))
    DASHBOARD_PASS = str(_cfg.get("dashboard", {}).get("password", "changeme"))
except Exception:
    DASHBOARD_PIN = "1234"
    DASHBOARD_USER = "admin"
    DASHBOARD_PASS = "changeme"

# Environment variables override config (so server credentials survive git pull)
DASHBOARD_USER = os.environ.get("DASH_USER", DASHBOARD_USER)
DASHBOARD_PASS = os.environ.get("DASH_PASS", DASHBOARD_PASS)

app.secret_key = secrets.token_hex(32)
app.permanent_session_lifetime = timedelta(hours=24)


# ── Auth ───────────────────────────────────────────────────────────

LOGIN_HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SMC Bot — Login</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,sans-serif;background:#1a1a2e;color:#e0e0e0;
     display:flex;justify-content:center;align-items:center;min-height:100vh}
.login-card{background:#16213e;border-radius:16px;padding:40px;width:340px;
            box-shadow:0 8px 32px rgba(0,0,0,0.4)}
h1{font-size:22px;text-align:center;margin-bottom:24px;color:#a0c4ff}
input{width:100%;padding:14px;margin:8px 0;background:#0f3460;border:1px solid #333;
      color:white;border-radius:8px;font-size:16px}
input:focus{outline:none;border-color:#4a9eff}
button{width:100%;padding:14px;margin-top:16px;background:#27ae60;color:white;
       border:none;border-radius:8px;font-size:18px;font-weight:bold;cursor:pointer}
button:hover{background:#2ecc71}
.error{color:#e74c3c;text-align:center;margin-top:12px;font-size:14px}
.subtitle{text-align:center;color:#666;font-size:13px;margin-bottom:20px}
</style></head><body>
<div class="login-card">
<h1>SMC Trading Bot</h1>
<p class="subtitle">Paper Trading Dashboard</p>
<form method="POST" action="/login">
<input type="text" name="username" placeholder="Username" autocomplete="username" required>
<input type="password" name="password" placeholder="Password" autocomplete="current-password" required>
<button type="submit">Login</button>
{% if error %}<p class="error">{{ error }}</p>{% endif %}
</form></div></body></html>"""


def login_required(f):
    """Decorator: redirect to /login if not authenticated."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("authenticated"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form.get("username", "")
        pwd = request.form.get("password", "")
        if user == DASHBOARD_USER and pwd == DASHBOARD_PASS:
            session.permanent = True
            session["authenticated"] = True
            return redirect(url_for("index"))
        return render_template_string(LOGIN_HTML, error="Invalid credentials"), 401
    return render_template_string(LOGIN_HTML, error=None)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ── Bot symbol -> asset class mapping (parsed once from logs) ─────
_BOT_MAP: dict[str, dict] = {}


def _verify_pin() -> str | None:
    """Check PIN from POST data. Returns error message or None if OK."""
    pin = request.form.get("pin", "")
    if pin != DASHBOARD_PIN:
        return "Invalid PIN"
    return None


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
    """Get bot uptime from paper_trading.log."""
    main_log = Path("paper_trading.log")
    if not main_log.exists():
        return "unknown"
    try:
        with open(main_log) as f:
            first_line = f.readline()
        m = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", first_line)
        if m:
            start = datetime.fromisoformat(m.group(1)).replace(tzinfo=timezone.utc)
            delta = datetime.now(timezone.utc) - start
            hours = int(delta.total_seconds() // 3600)
            mins = int((delta.total_seconds() % 3600) // 60)
            return f"{hours}h {mins}m"
    except Exception:
        pass
    return "unknown"


def _discover_bots() -> dict[str, dict]:
    """Discover all bots from state files."""
    global _BOT_MAP
    if _BOT_MAP and time.time() - _BOT_MAP.get("_ts", 0) < 30:
        return _BOT_MAP
    bots = {}
    for p in sorted(RESULTS_DIR.glob("bot_*_state.json")):
        tag = p.stem.replace("_state", "")
        try:
            with open(p) as f:
                state = json.load(f)
            # Infer asset class from log file first line
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
    bots["_ts"] = time.time()
    _BOT_MAP.clear()
    _BOT_MAP.update(bots)
    return bots


def _parse_log_entries(tag: str, max_lines: int = 500) -> list[dict]:
    """Parse recent log entries for a bot."""
    log_path = RESULTS_DIR / f"{tag}.log"
    if not log_path.exists():
        return []
    entries = []
    try:
        with open(log_path) as f:
            lines = f.readlines()[-max_lines:]
        for line in lines:
            entries.append(line.strip())
    except Exception:
        pass
    return entries


def _get_model_status() -> dict:
    """Parse paper_trading.log for model load status."""
    main_log = Path("paper_trading.log")
    models = {
        "entry_filter": {"loaded": False, "features": 0},
        "tp_optimizer": {"loaded": False, "features": 0},
        "be_manager": {"loaded": False, "features": 0},
    }
    if not main_log.exists():
        return models
    try:
        with open(main_log) as f:
            for line in f:
                m = re.search(r"Loaded model: models/rl_(\w+)\.pkl \((\d+) features\)", line)
                if m:
                    name = m.group(1)
                    if name in models:
                        models[name]["loaded"] = True
                        models[name]["features"] = int(m.group(2))
                if "RLBrainSuite initialized" in line:
                    break
    except Exception:
        pass
    return models


def _get_class_status() -> dict:
    """Parse paper_trading.log for asset class connection status."""
    main_log = Path("paper_trading.log")
    classes = {
        "crypto": {"status": "unknown", "adapter": "Binance", "errors": 0},
        "forex": {"status": "unknown", "adapter": "OANDA", "errors": 0},
        "stocks": {"status": "unknown", "adapter": "Alpaca", "errors": 0},
        "commodities": {"status": "unknown", "adapter": "OANDA", "errors": 0},
    }
    if not main_log.exists():
        return classes
    try:
        with open(main_log) as f:
            for line in f:
                if "Binance (crypto): connected" in line:
                    classes["crypto"]["status"] = "connected"
                elif "OANDA (forex+commodities): connected" in line:
                    classes["forex"]["status"] = "connected"
                    classes["commodities"]["status"] = "connected"
                elif "Alpaca (stocks): connected" in line:
                    classes["stocks"]["status"] = "connected"
                elif "disabled" in line.lower():
                    for cls in classes:
                        if cls in line.lower():
                            classes[cls]["status"] = "disabled"
        # Count recent errors per class
        with open(main_log) as f:
            lines = f.readlines()[-500:]
        for line in lines:
            if "[ERROR]" in line:
                for cls in classes:
                    if cls in line.lower() or classes[cls]["adapter"].lower() in line.lower():
                        classes[cls]["errors"] += 1
                # Alpaca SIP errors
                if "subscription does not permit" in line:
                    classes["stocks"]["errors"] += 1
    except Exception:
        pass
    return classes


def _aggregate_stats() -> dict:
    """Aggregate stats across all bots."""
    bots = _discover_bots()
    total_trades = 0
    total_wins = 0
    total_pnl = 0.0
    total_equity = 0.0
    active_positions = 0
    max_dd_pct = 0.0
    per_class = {}
    rl_accepted = 0
    rl_rejected = 0
    tp_adjusted = 0
    be_triggered = 0
    trade_log = []
    equity_points = []

    for tag, bot in bots.items():
        if tag.startswith("_"):
            continue
        ac = bot.get("asset_class", "unknown")
        trades = bot.get("trades", 0)
        wins = bot.get("wins", 0)
        pnl = bot.get("total_pnl", 0.0)
        equity = bot.get("equity", 0.0)
        peak = bot.get("peak_equity", 0.0)
        actives = bot.get("active_trades", [])

        total_trades += trades
        total_wins += wins
        total_pnl += pnl
        total_equity += equity
        active_positions += len(actives)

        if peak > 0:
            dd = (peak - equity) / peak * 100
            if dd > max_dd_pct:
                max_dd_pct = dd

        # Per-class aggregation
        if ac not in per_class:
            per_class[ac] = {"trades": 0, "wins": 0, "pnl": 0.0, "equity": 0.0}
        per_class[ac]["trades"] += trades
        per_class[ac]["wins"] += wins
        per_class[ac]["pnl"] += pnl
        per_class[ac]["equity"] += equity

        # Parse log for RL stats and trade entries
        log_lines = _parse_log_entries(tag, max_lines=300)
        last_rl_confidence = 0.0
        last_rl_filtered = False
        last_tp_adjusted = False
        last_be_level = 0.0
        for line in log_lines:
            if "XGB ACCEPT" in line:
                rl_accepted += 1
            elif "XGB REJECT" in line:
                rl_rejected += 1
            elif "RL TP adjusted" in line:
                tp_adjusted += 1
            elif "BE TRIGGERED" in line:
                be_triggered += 1

            # Track last XGB confidence for associating with OPEN entries
            if "XGB ACCEPT" in line:
                m_conf = re.search(r"conf=([\d.]+)", line)
                if m_conf:
                    last_rl_confidence = float(m_conf.group(1))
                    last_rl_filtered = True

            # Parse OPEN entries
            # Log format (live_multi_bot.py:2791): "OPEN [<STYLE>] <DIR> <SYM> @ ..."
            # Tier system killed 2026-04-19; only style group remains.
            if "OPEN [" in line:
                m = re.search(
                    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*"
                    r"OPEN \[(\w+)\] (\w+) (\S+) @ ([\d.]+).*"
                    r"SL=([\d.]+) TP=([\d.]+) RR=([\d.]+).*"
                    r"score=([\d.]+)",
                    line,
                )
                if m:
                    trade_entry = {
                        "time": m.group(1),
                        "style": m.group(2),
                        "direction": m.group(3),
                        "symbol": bot.get("symbol", tag),
                        "entry": float(m.group(5)),
                        "sl": float(m.group(6)),
                        "tp": float(m.group(7)),
                        "rr": float(m.group(8)),
                        "score": float(m.group(9)),
                        "type": "OPEN",
                        "asset_class": ac,
                        "rl_filtered": last_rl_filtered,
                        "rl_confidence": last_rl_confidence,
                        "tp_adjusted": last_tp_adjusted,
                        "be_level": last_be_level,
                    }
                    trade_log.append(trade_entry)
                    last_rl_filtered = False
                    last_rl_confidence = 0.0
                    last_tp_adjusted = False
                    last_be_level = 0.0

            # Track TP adjustments
            if "RL TP adjusted" in line:
                last_tp_adjusted = True

            # Track BE level set
            if "BE level" in line:
                m_be = re.search(r"BE level.*?=([\d.]+)", line)
                if m_be:
                    last_be_level = float(m_be.group(1))

            # Parse CLOSE entries
            if "CLOSE " in line and ("WIN" in line or "LOSS" in line):
                m = re.search(
                    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*"
                    r"CLOSE (\w+) (\w+) (\S+) @ ([\d.]+) . ([\d.]+).*"
                    r"pnl=([-\d.]+) equity=([\d.]+)",
                    line,
                )
                if m:
                    trade_log.append({
                        "time": m.group(1),
                        "outcome": m.group(2),
                        "direction": m.group(3),
                        "symbol": bot.get("symbol", tag),
                        "entry": float(m.group(5)),
                        "exit": float(m.group(6)),
                        "pnl": float(m.group(7)),
                        "equity": float(m.group(8)),
                        "type": "CLOSE",
                        "asset_class": ac,
                    })

        # Read equity CSV for chart data
        eq_path = RESULTS_DIR / f"{tag}_equity.csv"
        if eq_path.exists():
            try:
                with open(eq_path) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        equity_points.append({
                            "time": row.get("timestamp", ""),
                            "equity": float(row.get("equity", 0)),
                            "pnl": float(row.get("pnl", 0)),
                            "dd": float(row.get("drawdown_pct", 0)),
                            "tag": tag,
                        })
            except Exception:
                pass

    # Sort trade log by time (newest first), limit to 50
    trade_log.sort(key=lambda x: x.get("time", ""), reverse=True)
    trade_log = trade_log[:50]

    # Sort equity points by time for chart
    equity_points.sort(key=lambda x: x.get("time", ""))

    # Compute derived metrics
    wr = total_wins / total_trades * 100 if total_trades > 0 else 0
    rl_total = rl_accepted + rl_rejected
    rl_rate = rl_accepted / rl_total * 100 if rl_total > 0 else 0

    # Check bot process
    bot_running = _find_bot_pid() is not None

    # Compute uptime from main log
    uptime_str = _get_bot_uptime()

    # Kill switch status
    # Daily and weekly PnL would need time-windowed calculation
    # For now estimate from total PnL and uptime
    daily_pnl_pct = total_pnl / max(total_equity, 1) * 100 if total_equity > 0 else 0

    model_status = _get_model_status()
    class_status = _get_class_status()

    return {
        "bot_running": bot_running,
        "uptime": uptime_str,
        "model_status": model_status,
        "class_status": class_status,
        "total_equity": total_equity,
        "total_pnl": total_pnl,
        "total_trades": total_trades,
        "total_wins": total_wins,
        "win_rate": wr,
        "active_positions": active_positions,
        "max_dd_pct": max_dd_pct,
        "rl_accepted": rl_accepted,
        "rl_rejected": rl_rejected,
        "rl_rate": rl_rate,
        "tp_adjusted": tp_adjusted,
        "be_triggered": be_triggered,
        "per_class": per_class,
        "trade_log": trade_log,
        "equity_points": equity_points[-200:],  # Last 200 points for chart
        "daily_pnl_pct": daily_pnl_pct,
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }


def _get_ml_metrics() -> dict:
    """Build ML model health payload."""
    models_dir = Path("models")
    retrain_log = Path("backtest/results/rl/retrain_log.jsonl")
    rollback_watch = models_dir / ".rollback_watch"

    # Try to import SCHEMA_VERSION from features module
    feature_schema = None
    try:
        from features.feature_extractor import SCHEMA_VERSION as _SV
        feature_schema = _SV
    except Exception:
        pass

    model_files = {
        "entry_filter": "models/rl_entry_filter.pkl",
        "exit_classifier": "models/rl_exit_classifier.pkl",
        "dqn_exit_manager": "models/dqn_exit_manager.zip",
    }

    models_out = {}
    for name, rel_path in model_files.items():
        p = Path(rel_path)
        if p.exists():
            stat = p.stat()
            info: dict = {
                "path": rel_path,
                "exists": True,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            if feature_schema is not None:
                info["schema_version"] = feature_schema
        else:
            info = {"path": rel_path, "exists": False}
        models_out[name] = info

    # Last retrain entry
    last_retrain = None
    if retrain_log.exists():
        try:
            with open(retrain_log) as f:
                lines = [l.strip() for l in f if l.strip()]
            if lines:
                last_retrain = json.loads(lines[-1])
        except Exception:
            pass

    # Rollback watch file
    rollback_data = None
    if rollback_watch.exists():
        try:
            with open(rollback_watch) as f:
                rollback_data = f.read().strip() or None
        except Exception:
            pass

    return {
        "models": models_out,
        "last_retrain": last_retrain,
        "rollback_watch": rollback_data,
        "feature_schema": feature_schema,
    }


def _get_journal_recent() -> dict:
    """Return last 50 closed trades from journal SQLite DB plus aggregate stats."""
    db_path = Path("trade_journal/journal.db")
    empty = {"trades": [], "total_trades": 0, "stats": {"win_rate": 0.0, "avg_rr": 0.0, "total_pnl_pct": 0.0}}
    if not db_path.exists():
        return empty

    cols = [
        "trade_id", "symbol", "asset_class", "direction",
        "entry_time", "exit_time", "entry_price", "exit_price",
        "pnl_pct", "rr_actual", "rr_target", "outcome",
        "exit_reason", "bars_held", "tier", "score",
    ]

    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            # Total closed trades
            cur.execute("SELECT COUNT(*) FROM trades WHERE exit_time IS NOT NULL")
            total_trades = cur.fetchone()[0]

            # Last 50
            col_sql = ", ".join(c for c in cols if c in _journal_columns(cur))
            if not col_sql:
                return empty
            cur.execute(
                f"SELECT {col_sql} FROM trades WHERE exit_time IS NOT NULL "
                "ORDER BY exit_time DESC LIMIT 50"
            )
            rows = [dict(r) for r in cur.fetchall()]

            # Aggregate stats
            cur.execute(
                "SELECT COUNT(*) as n, SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END) as wins, "
                "AVG(rr_actual) as avg_rr, SUM(pnl_pct) as total_pnl "
                "FROM trades WHERE exit_time IS NOT NULL"
            )
            agg = cur.fetchone()
            n = agg[0] or 0
            wins = agg[1] or 0
            avg_rr = round(float(agg[2]), 3) if agg[2] is not None else 0.0
            total_pnl = round(float(agg[3]), 4) if agg[3] is not None else 0.0
            win_rate = round(wins / n, 4) if n > 0 else 0.0

    except Exception:
        return empty

    return {
        "trades": rows,
        "total_trades": total_trades,
        "stats": {
            "win_rate": win_rate,
            "avg_rr": avg_rr,
            "total_pnl_pct": total_pnl,
        },
    }


def _journal_columns(cur: sqlite3.Cursor) -> set:
    """Return the set of column names present in the trades table."""
    try:
        cur.execute("PRAGMA table_info(trades)")
        return {row[1] for row in cur.fetchall()}
    except Exception:
        return set()


def _get_trade_detail(trade_id: str) -> dict:
    """Return bar-by-bar data for one trade from the journal DB."""
    db_path = Path("trade_journal/journal.db")
    not_found = {"trade_id": trade_id, "bars": [], "metadata": {}}
    if not db_path.exists():
        return not_found

    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            # Metadata from trades table
            cur.execute(
                "SELECT * FROM trades WHERE trade_id = ? LIMIT 1", (trade_id,)
            )
            row = cur.fetchone()
            if row is None:
                return not_found
            metadata = dict(row)

            # Bar-by-bar data from trade_bars table (if it exists)
            bars = []
            try:
                cur.execute(
                    "SELECT * FROM trade_bars WHERE trade_id = ? ORDER BY bar_index ASC",
                    (trade_id,),
                )
                bars = [dict(r) for r in cur.fetchall()]
            except sqlite3.OperationalError:
                # Table doesn't exist yet
                pass

    except Exception:
        return not_found

    return {"trade_id": trade_id, "bars": bars, "metadata": metadata}


# ═══════════════════════════════════════════════════════════════════
#  Existing API Routes
# ═══════════════════════════════════════════════════════════════════

@app.route("/api/stats")
@login_required
def api_stats():
    return jsonify(_aggregate_stats())


@app.route("/api/ml-metrics")
@login_required
def api_ml_metrics():
    return jsonify(_get_ml_metrics())


@app.route("/api/journal/recent")
@login_required
def api_journal_recent():
    return jsonify(_get_journal_recent())


@app.route("/api/journal/trade/<trade_id>")
@login_required
def api_journal_trade(trade_id: str):
    return jsonify(_get_trade_detail(trade_id))


# ═══════════════════════════════════════════════════════════════════
#  Control Panel API Routes
# ═══════════════════════════════════════════════════════════════════

@app.route("/api/control/stop", methods=["POST"])
@login_required
def api_stop():
    """Kill the trading bot process."""
    err = _verify_pin()
    if err:
        return jsonify({"error": err}), 403

    pid = _find_bot_pid()
    if pid is None:
        return jsonify({"error": "Bot process not found"}), 404

    try:
        os.kill(pid, _signal.SIGTERM)
        # Wait up to 5 seconds for graceful shutdown
        for _ in range(10):
            time.sleep(0.5)
            try:
                os.kill(pid, 0)  # Check if still alive
            except OSError:
                return jsonify({"status": "stopped", "pid": pid})
        # Force kill if still running
        try:
            os.kill(pid, _signal.SIGKILL)
        except OSError:
            pass
        return jsonify({"status": "stopped", "pid": pid, "method": "SIGKILL"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/control/pause", methods=["POST"])
@login_required
def api_pause():
    """Pause new trades (existing trades continue)."""
    err = _verify_pin()
    if err:
        return jsonify({"error": err}), 403

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pause_path = RESULTS_DIR / ".pause_flag"
    pause_path.write_text(datetime.now(timezone.utc).isoformat())
    return jsonify({"status": "paused"})


@app.route("/api/control/resume", methods=["POST"])
@login_required
def api_resume():
    """Resume trading."""
    err = _verify_pin()
    if err:
        return jsonify({"error": err}), 403

    pause_path = RESULTS_DIR / ".pause_flag"
    if pause_path.exists():
        pause_path.unlink()
    return jsonify({"status": "running"})


@app.route("/api/control/toggle", methods=["POST"])
@login_required
def api_toggle_component():
    """Toggle a component on/off."""
    err = _verify_pin()
    if err:
        return jsonify({"error": err}), 403

    component = request.form.get("component", "")
    enabled = request.form.get("enabled", "true").lower() == "true"

    valid_components = {"entry_filter", "exit_classifier", "tp_optimizer", "be_manager"}
    if component not in valid_components:
        return jsonify({"error": f"Unknown component: {component}"}), 400

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    toggles_path = RESULTS_DIR / "component_toggles.json"
    toggles = {}
    if toggles_path.exists():
        try:
            with open(toggles_path) as f:
                toggles = json.load(f)
        except Exception:
            pass

    toggles[component] = enabled
    with open(toggles_path, "w") as f:
        json.dump(toggles, f, indent=2)

    return jsonify({"component": component, "enabled": enabled})


@app.route("/api/control/status")
@login_required
def api_control_status():
    """Get bot status without PIN."""
    pid = _find_bot_pid()
    paused = (RESULTS_DIR / ".pause_flag").exists()

    # Component toggles
    toggles_path = RESULTS_DIR / "component_toggles.json"
    toggles = {
        "entry_filter": True,
        "exit_classifier": True,
        "tp_optimizer": True,
        "be_manager": True,
    }
    if toggles_path.exists():
        try:
            with open(toggles_path) as f:
                saved = json.load(f)
            toggles.update(saved)
        except Exception:
            pass

    # Model info
    model_files = {
        "entry_filter": "models/rl_entry_filter.pkl",
        "exit_classifier": "models/rl_exit_classifier.pkl",
        "tp_optimizer": "models/rl_tp_optimizer.pkl",
        "be_manager": "models/rl_be_manager.pkl",
    }
    model_info = {}
    for name, rel_path in model_files.items():
        p = Path(rel_path)
        if p.exists():
            stat = p.stat()
            model_info[name] = {
                "exists": True,
                "modified": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
            }
        else:
            model_info[name] = {"exists": False}

    # Last retrain
    retrain_log = Path("backtest/results/rl/retrain_log.jsonl")
    last_retrain = None
    if retrain_log.exists():
        try:
            with open(retrain_log) as f:
                lines = [l.strip() for l in f if l.strip()]
            if lines:
                last_retrain = json.loads(lines[-1])
        except Exception:
            pass

    # Last trade from trade log
    last_trade = None
    bots = _discover_bots()
    for tag, bot in bots.items():
        if tag.startswith("_"):
            continue
        log_lines = _parse_log_entries(tag, max_lines=100)
        for line in reversed(log_lines):
            if "CLOSE " in line and ("WIN" in line or "LOSS" in line):
                m = re.search(
                    r"CLOSE (\w+) (\w+) (\S+) @ .* pnl=([-\d.]+)",
                    line,
                )
                if m:
                    last_trade = {
                        "outcome": m.group(1),
                        "direction": m.group(2),
                        "symbol": bot.get("symbol", tag),
                        "pnl": float(m.group(4)),
                    }
                    break
        if last_trade:
            break

    if pid is not None:
        state = "paused" if paused else "running"
    else:
        state = "stopped"

    return jsonify({
        "pid": pid,
        "state": state,
        "uptime": _get_bot_uptime(),
        "paused": paused,
        "toggles": toggles,
        "models": model_info,
        "last_retrain": last_retrain,
        "last_trade": last_trade,
    })


# ═══════════════════════════════════════════════════════════════════
#  Control Panel HTML
# ═══════════════════════════════════════════════════════════════════

@app.route("/")
@login_required
def index():
    return render_template_string(CONTROL_PANEL_HTML)


CONTROL_PANEL_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SMC Bot Control</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #1a1a2e; color: #e0e0e0; padding: 16px; min-height: 100vh; }
a { color: #58a6ff; text-decoration: none; }
a:hover { text-decoration: underline; }

/* Layout */
.container { max-width: 800px; margin: 0 auto; }

/* Header */
.header { display: flex; justify-content: space-between; align-items: center;
          flex-wrap: wrap; gap: 12px; margin-bottom: 20px; padding-bottom: 16px;
          border-bottom: 1px solid #30363d; }
.header h1 { font-size: 22px; color: #e0e0e0; font-weight: 700; }
.header-right { display: flex; align-items: center; gap: 12px; }
.status-indicator { display: flex; align-items: center; gap: 8px; font-size: 14px; font-weight: 600; }
.status-dot { display: inline-block; width: 12px; height: 12px; border-radius: 50%; }
.dot-green { background: #27ae60; box-shadow: 0 0 6px #27ae60; }
.dot-yellow { background: #f39c12; box-shadow: 0 0 6px #f39c12; }
.dot-red { background: #e74c3c; box-shadow: 0 0 6px #e74c3c; }
.meta-text { font-size: 12px; color: #8b949e; }

/* Cards */
.card { background: #16213e; border-radius: 12px; padding: 20px; margin: 12px 0;
        border: 1px solid #0f3460; }
.card h2 { font-size: 16px; margin-bottom: 16px; color: #a0a0c0;
           text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }

/* PIN input */
.pin-row { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }
.pin-label { font-size: 14px; color: #a0a0c0; }
.pin-input { padding: 12px; font-size: 24px; text-align: center; width: 130px;
             background: #0f3460; border: 1px solid #30363d; color: white;
             border-radius: 8px; letter-spacing: 8px; outline: none; }
.pin-input:focus { border-color: #58a6ff; }

/* Buttons */
.btn { display: inline-block; padding: 16px 32px; border: none; border-radius: 8px;
       font-size: 18px; font-weight: bold; cursor: pointer; width: 100%; margin: 6px 0;
       transition: opacity 0.2s, transform 0.1s; text-align: center; }
.btn:hover { opacity: 0.9; }
.btn:active { transform: scale(0.98); }
.btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
.btn-stop { background: #e74c3c; color: white; }
.btn-pause { background: #f39c12; color: white; }
.btn-resume { background: #27ae60; color: white; }
.btn-row { display: grid; grid-template-columns: 1fr; gap: 8px; }
@media(min-width: 500px) { .btn-row { grid-template-columns: 1fr 1fr 1fr; } }

/* Toggles */
.toggle-row { display: flex; justify-content: space-between; align-items: center;
              padding: 14px 0; border-bottom: 1px solid #0f3460; }
.toggle-row:last-child { border-bottom: none; }
.toggle-info { display: flex; flex-direction: column; gap: 2px; }
.toggle-name { font-size: 15px; font-weight: 600; }
.toggle-desc { font-size: 12px; color: #8b949e; }
.switch { position: relative; width: 52px; height: 28px; flex-shrink: 0; }
.switch input { opacity: 0; width: 0; height: 0; }
.slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0;
          background: #e74c3c; border-radius: 28px; transition: 0.3s; }
.slider:before { position: absolute; content: ""; height: 22px; width: 22px; left: 3px;
                 bottom: 3px; background: white; border-radius: 50%; transition: 0.3s; }
input:checked + .slider { background: #27ae60; }
input:checked + .slider:before { transform: translateX(24px); }

/* Status section */
.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
@media(min-width: 500px) { .stat-grid { grid-template-columns: 1fr 1fr 1fr; } }
.stat-item { background: #0f3460; border-radius: 8px; padding: 14px; text-align: center; }
.stat-value { font-size: 22px; font-weight: 700; margin-bottom: 4px; }
.stat-label { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; }
.green { color: #27ae60; }
.red { color: #e74c3c; }
.yellow { color: #f39c12; }
.blue { color: #58a6ff; }

/* Model info table */
.model-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.model-table th { text-align: left; padding: 8px 6px; color: #8b949e;
                  border-bottom: 1px solid #30363d; font-weight: 600; }
.model-table td { padding: 8px 6px; border-bottom: 1px solid #0f3460; }

/* Toast */
.toast { position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
         padding: 12px 24px; border-radius: 8px; font-size: 14px; font-weight: 600;
         z-index: 1000; opacity: 0; transition: opacity 0.3s; pointer-events: none; }
.toast.show { opacity: 1; }
.toast.success { background: #27ae60; color: white; }
.toast.error { background: #e74c3c; color: white; }

/* Nav link */
.nav-link { display: inline-block; margin-top: 12px; font-size: 13px; color: #58a6ff; }

/* Confirmation dialog */
.confirm-overlay { display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0;
                   background: rgba(0,0,0,0.7); z-index: 999; align-items: center;
                   justify-content: center; }
.confirm-overlay.active { display: flex; }
.confirm-box { background: #16213e; border: 1px solid #e74c3c; border-radius: 12px;
               padding: 24px; max-width: 400px; width: 90%; text-align: center; }
.confirm-box h3 { color: #e74c3c; margin-bottom: 12px; font-size: 18px; }
.confirm-box p { color: #a0a0c0; margin-bottom: 20px; font-size: 14px; }
.confirm-btns { display: flex; gap: 12px; }
.confirm-btns .btn { font-size: 14px; padding: 10px 20px; }
.btn-cancel { background: #30363d; color: #e0e0e0; }
</style>
</head>
<body>
<div class="container">
  <!-- Header -->
  <div class="header">
    <h1>SMC Trading Bot &mdash; Control Panel</h1>
    <a href="/logout" style="position:absolute;top:16px;right:16px;color:#888;font-size:13px;text-decoration:none;padding:6px 12px;border:1px solid #444;border-radius:6px;">Logout</a>
    <div class="header-right">
      <div class="status-indicator">
        <span class="status-dot dot-red" id="status-dot"></span>
        <span id="status-text">CHECKING...</span>
      </div>
      <div class="meta-text">
        PID: <span id="bot-pid">--</span> | Up: <span id="bot-uptime">--</span>
      </div>
    </div>
  </div>

  <!-- Emergency Controls -->
  <div class="card">
    <h2>Emergency Controls</h2>
    <div class="pin-row">
      <span class="pin-label">PIN:</span>
      <input type="password" class="pin-input" id="pin" maxlength="4" inputmode="numeric"
             pattern="[0-9]*" placeholder="----">
    </div>
    <div class="btn-row">
      <button class="btn btn-stop" id="btn-stop" onclick="confirmStop()">EMERGENCY STOP</button>
      <button class="btn btn-pause" id="btn-pause" onclick="doPause()">PAUSE ALL</button>
      <button class="btn btn-resume" id="btn-resume" onclick="doResume()">RESUME</button>
    </div>
  </div>

  <!-- Confirmation dialog for emergency stop -->
  <div class="confirm-overlay" id="confirm-overlay">
    <div class="confirm-box">
      <h3>CONFIRM EMERGENCY STOP</h3>
      <p>This will kill the live_multi_bot.py process. Existing positions will remain open but unmanaged.</p>
      <div class="confirm-btns">
        <button class="btn btn-cancel" onclick="cancelStop()">Cancel</button>
        <button class="btn btn-stop" onclick="doStop()">STOP BOT</button>
      </div>
    </div>
  </div>

  <!-- Component Toggles -->
  <div class="card">
    <h2>Component Toggles</h2>
    <div class="toggle-row">
      <div class="toggle-info">
        <span class="toggle-name">Entry Filter</span>
        <span class="toggle-desc">XGBoost entry prediction gate</span>
      </div>
      <label class="switch">
        <input type="checkbox" id="tog-entry_filter" checked onchange="doToggle('entry_filter', this.checked)">
        <span class="slider"></span>
      </label>
    </div>
    <div class="toggle-row">
      <div class="toggle-info">
        <span class="toggle-name">Exit Classifier</span>
        <span class="toggle-desc">ML early exit prediction</span>
      </div>
      <label class="switch">
        <input type="checkbox" id="tog-exit_classifier" checked onchange="doToggle('exit_classifier', this.checked)">
        <span class="slider"></span>
      </label>
    </div>
    <div class="toggle-row">
      <div class="toggle-info">
        <span class="toggle-name">TP Optimizer</span>
        <span class="toggle-desc">RL take-profit adjustment</span>
      </div>
      <label class="switch">
        <input type="checkbox" id="tog-tp_optimizer" checked onchange="doToggle('tp_optimizer', this.checked)">
        <span class="slider"></span>
      </label>
    </div>
    <div class="toggle-row">
      <div class="toggle-info">
        <span class="toggle-name">BE Manager</span>
        <span class="toggle-desc">Break-even level prediction</span>
      </div>
      <label class="switch">
        <input type="checkbox" id="tog-be_manager" checked onchange="doToggle('be_manager', this.checked)">
        <span class="slider"></span>
      </label>
    </div>
  </div>

  <!-- Status -->
  <div class="card">
    <h2>Live Status</h2>
    <div class="stat-grid">
      <div class="stat-item">
        <div class="stat-value" id="st-equity">--</div>
        <div class="stat-label">Equity</div>
      </div>
      <div class="stat-item">
        <div class="stat-value" id="st-trades">--</div>
        <div class="stat-label">Trades Today</div>
      </div>
      <div class="stat-item">
        <div class="stat-value" id="st-positions">--</div>
        <div class="stat-label">Active Positions</div>
      </div>
    </div>
  </div>

  <!-- Last Trade -->
  <div class="card">
    <h2>Last Trade</h2>
    <div id="last-trade-info" style="font-size:14px;color:#8b949e;">No trades recorded</div>
  </div>

  <!-- Model Versions -->
  <div class="card">
    <h2>ML Models</h2>
    <table class="model-table">
      <thead><tr><th>Model</th><th>Status</th><th>Size</th><th>Last Modified</th></tr></thead>
      <tbody id="model-tbody"></tbody>
    </table>
    <div id="retrain-info" style="margin-top:12px;font-size:12px;color:#8b949e;"></div>
  </div>

  <!-- Navigation to full dashboard -->
  <a class="nav-link" href="/api/stats" target="_blank">View raw stats JSON</a>
  &nbsp;|&nbsp;
  <a class="nav-link" href="/api/ml-metrics" target="_blank">View ML metrics JSON</a>
  &nbsp;|&nbsp;
  <a class="nav-link" href="/api/journal/recent" target="_blank">View journal JSON</a>
</div>

<!-- Toast notification -->
<div class="toast" id="toast"></div>

<script>
function getPin() {
  return document.getElementById('pin').value;
}

function showToast(msg, type) {
  var t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'toast show ' + type;
  setTimeout(function() { t.className = 'toast'; }, 3000);
}

function confirmStop() {
  if (!getPin()) { showToast('Enter PIN first', 'error'); return; }
  document.getElementById('confirm-overlay').classList.add('active');
}

function cancelStop() {
  document.getElementById('confirm-overlay').classList.remove('active');
}

function doStop() {
  document.getElementById('confirm-overlay').classList.remove('active');
  var fd = new FormData();
  fd.append('pin', getPin());
  fetch('/api/control/stop', { method: 'POST', body: fd })
    .then(function(r) { return r.json().then(function(d) { return { ok: r.ok, data: d }; }); })
    .then(function(res) {
      if (res.ok) { showToast('Bot stopped (PID ' + res.data.pid + ')', 'success'); }
      else { showToast(res.data.error || 'Failed', 'error'); }
      refreshStatus();
    })
    .catch(function() { showToast('Network error', 'error'); });
}

function doPause() {
  if (!getPin()) { showToast('Enter PIN first', 'error'); return; }
  var fd = new FormData();
  fd.append('pin', getPin());
  fetch('/api/control/pause', { method: 'POST', body: fd })
    .then(function(r) { return r.json().then(function(d) { return { ok: r.ok, data: d }; }); })
    .then(function(res) {
      if (res.ok) { showToast('Trading paused', 'success'); }
      else { showToast(res.data.error || 'Failed', 'error'); }
      refreshStatus();
    })
    .catch(function() { showToast('Network error', 'error'); });
}

function doResume() {
  if (!getPin()) { showToast('Enter PIN first', 'error'); return; }
  var fd = new FormData();
  fd.append('pin', getPin());
  fetch('/api/control/resume', { method: 'POST', body: fd })
    .then(function(r) { return r.json().then(function(d) { return { ok: r.ok, data: d }; }); })
    .then(function(res) {
      if (res.ok) { showToast('Trading resumed', 'success'); }
      else { showToast(res.data.error || 'Failed', 'error'); }
      refreshStatus();
    })
    .catch(function() { showToast('Network error', 'error'); });
}

function doToggle(component, enabled) {
  if (!getPin()) {
    showToast('Enter PIN first', 'error');
    document.getElementById('tog-' + component).checked = !enabled;
    return;
  }
  var fd = new FormData();
  fd.append('pin', getPin());
  fd.append('component', component);
  fd.append('enabled', enabled ? 'true' : 'false');
  fetch('/api/control/toggle', { method: 'POST', body: fd })
    .then(function(r) { return r.json().then(function(d) { return { ok: r.ok, data: d }; }); })
    .then(function(res) {
      if (res.ok) {
        showToast(res.data.component + ': ' + (res.data.enabled ? 'ON' : 'OFF'), 'success');
      } else {
        showToast(res.data.error || 'Failed', 'error');
        document.getElementById('tog-' + component).checked = !enabled;
      }
    })
    .catch(function() {
      showToast('Network error', 'error');
      document.getElementById('tog-' + component).checked = !enabled;
    });
}

function refreshStatus() {
  fetch('/api/control/status')
    .then(function(r) { return r.json(); })
    .then(function(d) {
      // Status indicator
      var dot = document.getElementById('status-dot');
      var txt = document.getElementById('status-text');
      if (d.state === 'running') {
        dot.className = 'status-dot dot-green';
        txt.textContent = 'RUNNING';
        txt.style.color = '#27ae60';
      } else if (d.state === 'paused') {
        dot.className = 'status-dot dot-yellow';
        txt.textContent = 'PAUSED';
        txt.style.color = '#f39c12';
      } else {
        dot.className = 'status-dot dot-red';
        txt.textContent = 'STOPPED';
        txt.style.color = '#e74c3c';
      }

      // PID + uptime
      document.getElementById('bot-pid').textContent = d.pid || '--';
      document.getElementById('bot-uptime').textContent = d.uptime || '--';

      // Toggle states
      if (d.toggles) {
        for (var key in d.toggles) {
          var el = document.getElementById('tog-' + key);
          if (el) el.checked = d.toggles[key];
        }
      }

      // Models table
      var tbody = document.getElementById('model-tbody');
      tbody.innerHTML = '';
      if (d.models) {
        for (var name in d.models) {
          var m = d.models[name];
          var statusCls = m.exists ? 'green' : 'red';
          var statusTxt = m.exists ? 'LOADED' : 'MISSING';
          var size = m.size_mb ? m.size_mb + ' MB' : '--';
          var modified = m.modified ? m.modified.replace('T', ' ').replace('Z', '') : '--';
          tbody.innerHTML += '<tr><td>' + name + '</td><td class="' + statusCls + '">' +
            statusTxt + '</td><td>' + size + '</td><td>' + modified + '</td></tr>';
        }
      }

      // Last retrain
      var ri = document.getElementById('retrain-info');
      if (d.last_retrain) {
        ri.textContent = 'Last retrain: ' + (d.last_retrain.timestamp || d.last_retrain.date || 'unknown');
      } else {
        ri.textContent = 'No retrain history';
      }

      // Last trade
      var lti = document.getElementById('last-trade-info');
      if (d.last_trade) {
        var lt = d.last_trade;
        var pnlCls = lt.pnl >= 0 ? 'green' : 'red';
        var pnlSign = lt.pnl >= 0 ? '+' : '';
        lti.innerHTML = '<span style="font-size:15px;font-weight:600;">' + lt.symbol + '</span>' +
          ' &mdash; ' + lt.direction.toUpperCase() +
          ' &mdash; <span class="' + pnlCls + '">' + pnlSign + '$' + Number(lt.pnl).toFixed(2) + '</span>' +
          ' (<span class="' + (lt.outcome === 'WIN' ? 'green' : 'red') + '">' + lt.outcome + '</span>)';
      }
    })
    .catch(function(e) { console.error('Status refresh failed:', e); });

  // Also fetch aggregate stats for equity/trades/positions
  fetch('/api/stats')
    .then(function(r) { return r.json(); })
    .then(function(d) {
      document.getElementById('st-equity').textContent = '$' + Number(d.total_equity).toFixed(2);
      document.getElementById('st-trades').textContent = d.total_trades;
      document.getElementById('st-positions').textContent = d.active_positions;
    })
    .catch(function() {});
}

// Initial load + auto-refresh every 30s
refreshStatus();
setInterval(refreshStatus, 30000);
</script>
</body>
</html>"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    # Override from config if available
    try:
        with open("config/default_config.yaml") as f:
            cfg = yaml.safe_load(f)
        dash_cfg = cfg.get("dashboard", {})
        host = args.host if args.host != "0.0.0.0" else dash_cfg.get("host", "0.0.0.0")
        port = args.port if args.port != 8080 else dash_cfg.get("port", 8080)
    except Exception:
        host = args.host
        port = args.port
    print(f"Dashboard starting on http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
