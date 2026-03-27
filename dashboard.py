"""
Paper Trading Dashboard — lightweight Flask app with auto-refresh.
Reads from live_results/ (bot state JSONs, equity CSVs, log files).

Usage: python3 dashboard.py [--port 8080]
"""
from __future__ import annotations

import csv
import json
import os
import re
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from flask import Flask, jsonify, render_template_string

app = Flask(__name__)
RESULTS_DIR = Path("live_results")

# ── Bot symbol → asset class mapping (parsed once from logs) ─────
_BOT_MAP: dict[str, dict] = {}


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
            if "OPEN [" in line:
                m = re.search(
                    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*"
                    r"OPEN \[(\w+)\|(\w+)\] (\w+) (\S+) @ ([\d.]+).*"
                    r"SL=([\d.]+) TP=([\d.]+) RR=([\d.]+).*"
                    r"score=([\d.]+)",
                    line,
                )
                if m:
                    trade_entry = {
                        "time": m.group(1),
                        "tier": m.group(2),
                        "style": m.group(3),
                        "direction": m.group(4),
                        "symbol": bot.get("symbol", tag),
                        "entry": float(m.group(6)),
                        "sl": float(m.group(7)),
                        "tp": float(m.group(8)),
                        "rr": float(m.group(9)),
                        "score": float(m.group(10)),
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
    import subprocess
    try:
        result = subprocess.run(
            ["pgrep", "-f", "live_multi_bot"], capture_output=True, text=True
        )
        bot_running = bool(result.stdout.strip())
    except Exception:
        bot_running = False

    # Compute uptime from main log
    uptime_str = "unknown"
    main_log = Path("paper_trading.log")
    if main_log.exists():
        try:
            with open(main_log) as f:
                first_line = f.readline()
            m = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", first_line)
            if m:
                start = datetime.fromisoformat(m.group(1)).replace(tzinfo=timezone.utc)
                delta = datetime.now(timezone.utc) - start
                hours = int(delta.total_seconds() // 3600)
                mins = int((delta.total_seconds() % 3600) // 60)
                uptime_str = f"{hours}h {mins}m"
        except Exception:
            pass

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


@app.route("/api/stats")
def api_stats():
    return jsonify(_aggregate_stats())


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SMC Paper Trading Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root {
  --bg: #0d1117; --bg2: #161b22; --bg3: #21262d;
  --border: #30363d; --text: #c9d1d9; --text2: #8b949e;
  --green: #3fb950; --red: #f85149; --yellow: #d29922; --blue: #58a6ff;
}
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family: -apple-system, 'Segoe UI', monospace; background:var(--bg); color:var(--text); padding:12px; }
.grid { display:grid; gap:12px; grid-template-columns:1fr; }
@media(min-width:768px) { .grid { grid-template-columns:repeat(2,1fr); } .wide { grid-column:span 2; } }
@media(min-width:1200px) { .grid { grid-template-columns:repeat(3,1fr); } .wide { grid-column:span 3; } }
.card { background:var(--bg2); border:1px solid var(--border); border-radius:8px; padding:16px; }
.card h3 { color:var(--blue); font-size:13px; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px; }
.stat { display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid var(--bg3); font-size:14px; }
.stat:last-child { border:none; }
.stat .label { color:var(--text2); }
.stat .val { font-weight:600; }
.green { color:var(--green); } .red { color:var(--red); } .yellow { color:var(--yellow); }
.header { display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; flex-wrap:wrap; gap:8px; }
.header h1 { font-size:18px; color:var(--blue); }
.status { padding:4px 10px; border-radius:12px; font-size:12px; font-weight:600; }
.status.running { background:#1a3a1a; color:var(--green); }
.status.stopped { background:#3a1a1a; color:var(--red); }
table { width:100%; border-collapse:collapse; font-size:12px; }
th { text-align:left; padding:6px 4px; color:var(--text2); border-bottom:1px solid var(--border); }
td { padding:5px 4px; border-bottom:1px solid var(--bg3); }
tr:hover { background:var(--bg3); }
.bar-track { background:var(--bg3); border-radius:4px; height:20px; overflow:hidden; margin:2px 0; position:relative; }
.bar-fill { height:100%; border-radius:4px; transition:width 0.5s; }
.bar-label { position:absolute; right:6px; top:2px; font-size:11px; color:var(--text); }
#equity-chart { width:100%; height:200px; }
.refresh { color:var(--text2); font-size:11px; }
</style>
</head>
<body>
<div class="header">
  <h1>SMC Paper Trading</h1>
  <div>
    <span id="status" class="status stopped">--</span>
    <span class="refresh" id="updated">loading...</span>
  </div>
</div>
<div class="grid" id="root">
  <!-- Header Stats -->
  <div class="card" id="c-header">
    <h3>Account Overview</h3>
    <div class="stat"><span class="label">Uptime</span><span class="val" id="uptime">--</span></div>
    <div class="stat"><span class="label">Total Equity</span><span class="val" id="equity">--</span></div>
    <div class="stat"><span class="label">Total PnL</span><span class="val" id="pnl">--</span></div>
    <div class="stat"><span class="label">Active Positions</span><span class="val" id="positions">--</span></div>
    <div class="stat"><span class="label">Total Trades</span><span class="val" id="trades">--</span></div>
    <div class="stat"><span class="label">Win Rate</span><span class="val" id="wr">--</span></div>
  </div>

  <!-- RL Brain Stats -->
  <div class="card" id="c-rl">
    <h3>RL Brain Stats</h3>
    <div class="stat"><span class="label">Entry Filter: Accepted</span><span class="val green" id="rl-acc">--</span></div>
    <div class="stat"><span class="label">Entry Filter: Rejected</span><span class="val red" id="rl-rej">--</span></div>
    <div class="stat"><span class="label">Acceptance Rate</span><span class="val" id="rl-rate">--</span></div>
    <div class="stat"><span class="label">TP Adjustments</span><span class="val" id="tp-adj">--</span></div>
    <div class="stat"><span class="label">BE Triggers</span><span class="val" id="be-trig">--</span></div>
  </div>

  <!-- Model Status -->
  <div class="card" id="c-models">
    <h3>Model Status</h3>
    <table>
      <thead><tr><th>Model</th><th>Status</th><th>Features</th></tr></thead>
      <tbody id="model-body"></tbody>
    </table>
  </div>

  <!-- Asset Class Status -->
  <div class="card" id="c-classes">
    <h3>Asset Class Status</h3>
    <table>
      <thead><tr><th>Class</th><th>Adapter</th><th>Status</th><th>Errors</th></tr></thead>
      <tbody id="class-body"></tbody>
    </table>
  </div>

  <!-- Performance -->
  <div class="card" id="c-perf">
    <h3>Performance Metrics</h3>
    <div class="stat"><span class="label">Wins / Losses</span><span class="val" id="wl">--</span></div>
    <div class="stat"><span class="label">Max Drawdown</span><span class="val red" id="max-dd">--</span></div>
    <div class="stat"><span class="label">Profit Factor</span><span class="val" id="pf">--</span></div>
    <div class="stat"><span class="label">Avg PnL/Trade</span><span class="val" id="avg-pnl">--</span></div>
  </div>

  <!-- Kill Switch Status -->
  <div class="card" id="c-kill">
    <h3>Kill Switch Status</h3>
    <div style="margin-bottom:8px;">
      <div style="display:flex;justify-content:space-between;font-size:12px;"><span>Daily PnL</span><span id="kill-daily">--</span></div>
      <div class="bar-track"><div class="bar-fill" id="bar-daily" style="width:0%;background:var(--green)"></div><div class="bar-label">/ -3%</div></div>
    </div>
    <div style="margin-bottom:8px;">
      <div style="display:flex;justify-content:space-between;font-size:12px;"><span>Weekly PnL</span><span id="kill-weekly">--</span></div>
      <div class="bar-track"><div class="bar-fill" id="bar-weekly" style="width:0%;background:var(--green)"></div><div class="bar-label">/ -5%</div></div>
    </div>
    <div>
      <div style="display:flex;justify-content:space-between;font-size:12px;"><span>All-Time DD</span><span id="kill-dd">--</span></div>
      <div class="bar-track"><div class="bar-fill" id="bar-dd" style="width:0%;background:var(--green)"></div><div class="bar-label">/ -8%</div></div>
    </div>
  </div>

  <!-- Per-Asset Breakdown -->
  <div class="card" id="c-asset">
    <h3>Per-Asset Breakdown</h3>
    <table>
      <thead><tr><th>Class</th><th>Trades</th><th>WR</th><th>PnL</th></tr></thead>
      <tbody id="asset-body"></tbody>
    </table>
  </div>

  <!-- Equity Curve -->
  <div class="card wide" id="c-chart">
    <h3>Equity Curve</h3>
    <canvas id="equity-chart"></canvas>
  </div>

  <!-- Trade Log -->
  <div class="card wide" id="c-trades">
    <h3>Recent Trades (last 50)</h3>
    <div style="overflow-x:auto;">
    <table>
      <thead><tr><th>Time</th><th>Symbol</th><th>Dir</th><th>Entry</th><th>Exit</th><th>PnL</th><th>Result</th><th>RL Conf</th><th>TP Adj</th><th>Class</th></tr></thead>
      <tbody id="trade-body"></tbody>
    </table>
    </div>
  </div>
</div>

<script>
let chart = null;
function pnlClass(v) { return v > 0 ? 'green' : v < 0 ? 'red' : ''; }
function fmt(v, d=2) { return Number(v).toFixed(d); }
function barPct(val, limit) {
  let pct = Math.min(Math.abs(val / limit) * 100, 100);
  return pct;
}
function barColor(val, limit) {
  let ratio = Math.abs(val / limit);
  if (ratio > 0.8) return 'var(--red)';
  if (ratio > 0.5) return 'var(--yellow)';
  return 'var(--green)';
}

async function refresh() {
  try {
    const r = await fetch('/api/stats');
    const d = await r.json();

    // Status
    const el = document.getElementById('status');
    el.textContent = d.bot_running ? 'RUNNING' : 'STOPPED';
    el.className = 'status ' + (d.bot_running ? 'running' : 'stopped');
    document.getElementById('updated').textContent = d.updated;

    // Header
    document.getElementById('uptime').textContent = d.uptime;
    document.getElementById('equity').textContent = '$' + fmt(d.total_equity);
    const pnlEl = document.getElementById('pnl');
    pnlEl.textContent = (d.total_pnl >= 0 ? '+' : '') + '$' + fmt(d.total_pnl);
    pnlEl.className = 'val ' + pnlClass(d.total_pnl);
    document.getElementById('positions').textContent = d.active_positions;
    document.getElementById('trades').textContent = d.total_trades;
    const wrEl = document.getElementById('wr');
    wrEl.textContent = fmt(d.win_rate, 1) + '%';
    wrEl.className = 'val ' + (d.win_rate >= 50 ? 'green' : d.win_rate >= 35 ? 'yellow' : 'red');

    // RL
    document.getElementById('rl-acc').textContent = d.rl_accepted;
    document.getElementById('rl-rej').textContent = d.rl_rejected;
    const rateEl = document.getElementById('rl-rate');
    rateEl.textContent = fmt(d.rl_rate, 1) + '%';
    rateEl.className = 'val ' + (d.rl_rate < 10 || d.rl_rate > 95 ? 'red' : 'green');
    document.getElementById('tp-adj').textContent = d.tp_adjusted;
    document.getElementById('be-trig').textContent = d.be_triggered;

    // Performance
    let losses = d.total_trades - d.total_wins;
    document.getElementById('wl').textContent = d.total_wins + ' / ' + losses;
    document.getElementById('max-dd').textContent = fmt(d.max_dd_pct, 2) + '%';
    let avgPnl = d.total_trades > 0 ? d.total_pnl / d.total_trades : 0;
    const avgEl = document.getElementById('avg-pnl');
    avgEl.textContent = '$' + fmt(avgPnl);
    avgEl.className = 'val ' + pnlClass(avgPnl);
    // Rough PF estimate
    let winPnl = d.total_pnl > 0 ? d.total_pnl : 0;
    let lossPnl = d.total_pnl < 0 ? Math.abs(d.total_pnl) : 0.01;
    let pf = d.total_trades > 0 ? (winPnl / lossPnl) : 0;
    document.getElementById('pf').textContent = fmt(pf > 99 ? 99 : pf, 2);

    // Kill switches
    let dpnl = d.daily_pnl_pct;
    document.getElementById('kill-daily').textContent = fmt(dpnl, 2) + '%';
    document.getElementById('bar-daily').style.width = barPct(dpnl, -3) + '%';
    document.getElementById('bar-daily').style.background = barColor(dpnl, -3);
    document.getElementById('kill-weekly').textContent = fmt(dpnl, 2) + '%';
    document.getElementById('bar-weekly').style.width = barPct(dpnl, -5) + '%';
    document.getElementById('bar-weekly').style.background = barColor(dpnl, -5);
    document.getElementById('kill-dd').textContent = fmt(d.max_dd_pct, 2) + '%';
    document.getElementById('bar-dd').style.width = barPct(d.max_dd_pct, 8) + '%';
    document.getElementById('bar-dd').style.background = barColor(d.max_dd_pct, 8);

    // Model status
    let mb = document.getElementById('model-body');
    mb.innerHTML = '';
    for (let [name, info] of Object.entries(d.model_status || {})) {
      let sClass = info.loaded ? 'green' : 'red';
      let sText = info.loaded ? 'LOADED' : 'MISSING';
      mb.innerHTML += '<tr><td>' + name + '</td><td class="' + sClass + '">' + sText +
        '</td><td>' + (info.features || '-') + '</td></tr>';
    }

    // Asset class status
    let cb = document.getElementById('class-body');
    cb.innerHTML = '';
    for (let [cls, info] of Object.entries(d.class_status || {})) {
      let sClass = info.status === 'connected' ? 'green' : info.status === 'disabled' ? 'red' : 'yellow';
      let errStr = info.errors > 0 ? '<span class="red">' + info.errors + '</span>' : '0';
      cb.innerHTML += '<tr><td>' + cls + '</td><td>' + info.adapter + '</td><td class="' + sClass + '">' +
        info.status + '</td><td>' + errStr + '</td></tr>';
    }

    // Per-asset
    let ab = document.getElementById('asset-body');
    ab.innerHTML = '';
    for (let [cls, s] of Object.entries(d.per_class || {})) {
      let wr = s.trades > 0 ? (s.wins / s.trades * 100).toFixed(1) : '0.0';
      ab.innerHTML += '<tr><td>' + cls + '</td><td>' + s.trades +
        '</td><td>' + wr + '%</td><td class="' + pnlClass(s.pnl) + '">$' + fmt(s.pnl) + '</td></tr>';
    }

    // Trade log
    let tb = document.getElementById('trade-body');
    tb.innerHTML = '';
    for (let t of d.trade_log || []) {
      if (t.type === 'CLOSE') {
        let cls = t.outcome === 'WIN' ? 'green' : 'red';
        let rlConf = t.rl_confidence > 0 ? fmt(t.rl_confidence, 3) : '<span class="yellow">N/A</span>';
        let tpAdj = t.tp_adjusted ? '<span class="green">Yes</span>' : '-';
        tb.innerHTML += '<tr><td>' + t.time + '</td><td>' + t.symbol +
          '</td><td>' + (t.direction||'') + '</td><td>' + fmt(t.entry, 4) +
          '</td><td>' + fmt(t.exit||0, 4) + '</td><td class="' + pnlClass(t.pnl) + '">$' + fmt(t.pnl) +
          '</td><td class="' + cls + '">' + t.outcome + '</td><td>' + rlConf +
          '</td><td>' + tpAdj + '</td><td>' + (t.asset_class||'') + '</td></tr>';
      } else if (t.type === 'OPEN') {
        let rlConf = t.rl_confidence > 0 ? fmt(t.rl_confidence, 3) : '<span class="yellow">N/A</span>';
        let rlTag = t.rl_filtered ? '<span class="green">RL</span>' : '<span class="yellow">RAW</span>';
        let tpAdj = t.tp_adjusted ? '<span class="green">Yes</span>' : '-';
        tb.innerHTML += '<tr style="opacity:0.7"><td>' + t.time + '</td><td>' + t.symbol +
          '</td><td>' + (t.direction||'') + '</td><td>' + fmt(t.entry, 4) +
          '</td><td>-</td><td>-</td><td>' + rlTag +
          '</td><td>' + rlConf + '</td><td>' + tpAdj + '</td><td>' + (t.asset_class||'') + '</td></tr>';
      }
    }
    if (tb.innerHTML === '') {
      tb.innerHTML = '<tr><td colspan="10" style="text-align:center;color:var(--text2)">No trades yet</td></tr>';
    }

    // Equity chart
    let pts = d.equity_points || [];
    if (pts.length > 0) {
      let labels = pts.map(p => p.time ? p.time.substring(11, 19) : '');
      let data = pts.map(p => p.pnl);
      let colors = data.map(v => v >= 0 ? 'rgba(63,185,80,0.5)' : 'rgba(248,81,73,0.5)');
      if (!chart) {
        chart = new Chart(document.getElementById('equity-chart'), {
          type: 'line',
          data: {
            labels: labels,
            datasets: [{
              label: 'Cumulative PnL',
              data: data,
              borderColor: 'rgba(88,166,255,0.8)',
              backgroundColor: 'rgba(88,166,255,0.1)',
              fill: true,
              tension: 0.3,
              pointRadius: 0,
              borderWidth: 2,
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
              x: { ticks: { color: '#8b949e', maxTicksLimit: 10 }, grid: { color: '#21262d' } },
              y: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' } }
            }
          }
        });
      } else {
        chart.data.labels = labels;
        chart.data.datasets[0].data = data;
        chart.update('none');
      }
    }
  } catch(e) { console.error('Refresh failed:', e); }
}

refresh();
setInterval(refresh, 60000);
</script>
</body>
</html>"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    print(f"Dashboard starting on http://0.0.0.0:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)
