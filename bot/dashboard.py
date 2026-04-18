"""Rich-based TUI dashboard helpers.

Extracted from live_multi_bot.py during Phase 3 restructure
(2026-04-18). Pure presentation layer: takes a list of bot-like
objects + runtime state and renders a ``Layout``. No direct
dependency on PaperBot internals other than the subset of
attributes listed in ``BotSummary`` below.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Protocol

from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from paper_grid import PaperGrid


# Matches WS_GROUP_SIZE in live_multi_bot.py — duplicated here to keep
# the dashboard module import-free of runtime constants. If this ever
# needs to change, update both sites.
WS_GROUP_SIZE = 10


class BotSummary(Protocol):
    """Attributes the dashboard reads from a bot instance.

    Anything that quacks like this works — no import of PaperBot.
    """

    asset_class: str
    total_pnl: float
    trades: int

    def summary_dict(self) -> dict[str, Any]: ...


def _pnl_color(value: float) -> str:
    """Return rich markup colour for a PnL value."""
    if value > 0:
        return "green"
    elif value < 0:
        return "red"
    return "white"


def _format_uptime(start: datetime) -> str:
    """Human-readable uptime string."""
    delta = datetime.now(timezone.utc) - start
    total_sec = int(delta.total_seconds())
    hours, remainder = divmod(total_sec, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}h {minutes:02d}m {seconds:02d}s"


def _build_bot_table(
    title: str,
    rows: list[dict[str, Any]],
    style: str,
) -> Table:
    """Build a Rich Table for a set of bot summaries."""
    table = Table(
        title=title,
        title_style=f"bold {style}",
        border_style=style,
        show_lines=False,
        expand=True,
    )
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Bot-ID", style="cyan", width=9)
    table.add_column("Symbol", style="bright_yellow", width=18)
    table.add_column("Class", style="dim", width=7)
    table.add_column("PnL", justify="right", width=14)
    table.add_column("Return%", justify="right", width=10)
    table.add_column("Trades", justify="right", width=10)
    table.add_column("Winrate", justify="right", width=10)
    table.add_column("DD%", justify="right", width=9)
    table.add_column("Open", justify="right", width=7)

    for i, r in enumerate(rows, 1):
        pnl_c = _pnl_color(r["pnl"])
        ret_c = _pnl_color(r["return_pct"])
        table.add_row(
            str(i),
            r["bot"],
            r.get("symbol", ""),
            r.get("asset_class", "crypto")[:6],
            f"[{pnl_c}]{r['pnl']:+,.2f}[/{pnl_c}]",
            f"[{ret_c}]{r['return_pct']:+.2f}%[/{ret_c}]",
            str(r["trades"]),
            f"{r['winrate']:.1f}%",
            f"{r['drawdown_pct']:.2f}%",
            str(r["open_pos"]),
        )

    return table


def build_dashboard(
    bots: list[BotSummary],
    ws_status: dict[str, str],
    start_time: datetime,
    active_symbols: list[str],
    total_equity: float = 0.0,
    paper_grid: "PaperGrid | None" = None,
) -> Layout:
    """
    Build the complete Rich Layout for the live dashboard.

    Returns a Layout containing:
      - Header panel (title, total equity, uptime)
      - Top 20 bots table
      - Worst 20 bots table
      - WebSocket status panel
    """
    all_summaries = sorted(
        [b.summary_dict() for b in bots],
        key=lambda r: r["pnl"],
        reverse=True,
    )

    total_pnl = sum(b.total_pnl for b in bots)
    total_trades = sum(b.trades for b in bots)
    uptime = _format_uptime(start_time)
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    class_counts: dict[str, int] = {}
    for b in bots:
        class_counts[b.asset_class] = class_counts.get(b.asset_class, 0) + 1
    class_str = " | ".join(f"{ac}: {cnt}" for ac, cnt in sorted(class_counts.items()))

    eq_color = _pnl_color(total_pnl)
    header_text = Text.from_markup(
        f"[bold cyan]📊 SMC MULTI-ASSET LIVE TRADING DASHBOARD[/bold cyan]\n"
        f"[dim]{now_str}[/dim]  ·  Uptime: [bold]{uptime}[/bold]  ·  "
        f"Bots: [bold]{len(bots)}[/bold] ({class_str})\n"
        f"Total Equity: [bold]{total_equity:,.2f}[/bold]  ·  "
        f"Total PnL: [bold {eq_color}]{total_pnl:+,.2f}[/bold {eq_color}]  ·  "
        f"Total Trades: [bold]{total_trades}[/bold]",
    )
    header_panel = Panel(
        header_text,
        title="[bold white]HEADER[/bold white]",
        border_style="bright_blue",
    )

    top_20 = all_summaries[:20]
    worst_20 = list(reversed(all_summaries[-20:])) if len(all_summaries) > 20 else list(reversed(all_summaries))

    top_table = _build_bot_table("🏆  TOP 20 BOTS", top_20, "green")
    worst_table = _build_bot_table("📉  WORST 20 BOTS", worst_20, "red")

    statuses = list(ws_status.values())
    n_connected = statuses.count("connected")
    n_reconnecting = sum(1 for s in statuses if s.startswith("reconnecting"))
    n_disconnected = statuses.count("disconnected")

    if n_disconnected > 0:
        global_label = f"[bold red]⛔ DISCONNECTED ({n_disconnected})[/bold red]"
    elif n_reconnecting > 0:
        global_label = f"[bold yellow]🔄 RECONNECTING ({n_reconnecting})[/bold yellow]"
    else:
        global_label = f"[bold green]✅ ALL CONNECTED ({n_connected})[/bold green]"

    ws_lines = [f"Global: {global_label}\n"]

    sorted_keys = sorted(ws_status.keys())
    groups: dict[int, list[str]] = {}
    for i, sym in enumerate(sorted_keys):
        idx = i // WS_GROUP_SIZE
        groups.setdefault(idx, []).append(sym)

    for gid, syms in sorted(groups.items()):
        group_statuses = [ws_status.get(s, "unknown") for s in syms]
        gc = sum(1 for gs in group_statuses if gs == "connected")
        gr = sum(1 for gs in group_statuses if gs.startswith("reconnecting"))
        gd = sum(1 for gs in group_statuses if gs == "disconnected")

        if gd > 0:
            gs_label = f"[red]⛔ {gc}/{len(syms)} connected, {gd} disconnected[/red]"
        elif gr > 0:
            gs_label = f"[yellow]🔄 {gc}/{len(syms)} connected, {gr} reconnecting[/yellow]"
        else:
            gs_label = f"[green]✅ {gc}/{len(syms)} connected[/green]"

        ws_lines.append(f"  Group {gid + 1} ({len(syms)} symbols): {gs_label}")

    ws_panel = Panel(
        Text.from_markup("\n".join(ws_lines)),
        title="[bold white]WEBSOCKET STATUS[/bold white]",
        border_style="bright_blue",
    )

    grid_panel = None
    if paper_grid is not None:
        grid_rows = paper_grid.dashboard_data()
        grid_table = Table(title="Paper Grid (Top 10 Variants)", expand=True)
        grid_table.add_column("Variant", style="cyan", no_wrap=True)
        grid_table.add_column("PnL", justify="right")
        grid_table.add_column("PnL%", justify="right")
        grid_table.add_column("DD%", justify="right")
        grid_table.add_column("Trades", justify="right")
        grid_table.add_column("WR%", justify="right")
        grid_table.add_column("PF", justify="right")
        grid_table.add_column("BE%", justify="right")
        grid_table.add_column("Open", justify="right")
        grid_table.add_column("Params", style="dim")

        for row in grid_rows[:10]:
            pnl_c = "green" if row["pnl"] >= 0 else "red"
            dd_c = "red" if row["dd_pct"] < -5 else "yellow" if row["dd_pct"] < -2 else "green"
            grid_table.add_row(
                row["name"],
                f"[{pnl_c}]${row['pnl']:+,.0f}[/{pnl_c}]",
                f"[{pnl_c}]{row['pnl_pct']:+.1f}%[/{pnl_c}]",
                f"[{dd_c}]{row['dd_pct']:.1f}%[/{dd_c}]",
                str(row["trades"]),
                f"{row['wr_real']:.0f}%" if row["trades"] > 0 else "-",
                f"{row['pf_real']:.1f}" if row["trades"] > 0 else "-",
                f"{row['be_rate']:.0f}%" if row["trades"] > 0 else "-",
                str(row["open"]),
                f"A={row['align']:.2f} RR={row['rr']:.1f} L={row['lev']} R={row['risk']:.1f}%",
            )

        grid_panel = Panel(grid_table, border_style="bright_magenta")

    layout = Layout()
    if grid_panel is not None:
        layout.split_column(
            Layout(header_panel, name="header", size=6),
            Layout(name="tables"),
            Layout(grid_panel, name="grid", size=14),
            Layout(ws_panel, name="status", size=3 + len(groups) + 2),
        )
    else:
        layout.split_column(
            Layout(header_panel, name="header", size=6),
            Layout(name="tables"),
            Layout(ws_panel, name="status", size=3 + len(groups) + 2),
        )
    layout["tables"].split_row(
        Layout(top_table, name="top"),
        Layout(worst_table, name="worst"),
    )

    return layout
