"""
Rich Terminal Dashboard — Real-time bot status display.
Easy to interpret, updates in-place.
"""

import time
from typing import Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from core.market import Position, EdgeSignal
from execution.risk_manager import RiskManager
from dashboard.logger import TradeLogger


console = Console()


def _heat_color(level: str) -> str:
    return {"GREEN": "green", "YELLOW": "yellow", "ORANGE": "dark_orange", "RED": "red"}.get(level, "white")


def _pnl_color(pnl: float) -> str:
    return "green" if pnl >= 0 else "red"


def build_status_panel(risk: RiskManager, scan_count: int, uptime_seconds: float) -> Panel:
    """Build the main status panel."""
    heat = risk.heat_level
    color = _heat_color(heat)

    lines = [
        f"[bold]Bankroll:[/] ${risk.bankroll:.2f}  |  "
        f"[bold]P&L:[/] [{_pnl_color(risk.daily_pnl)}]${risk.daily_pnl:+.2f}[/]  |  "
        f"[bold]DD:[/] [{color}]{risk.drawdown:.1%} [{heat}][/]",
        "",
        f"[bold]Trades:[/] {risk.trade_count}/{risk.bankroll:.0f}  |  "
        f"[bold]Win Rate:[/] {risk.win_rate:.0%} ({len(risk.trade_history)} trades)  |  "
        f"[bold]Avg P&L:[/] [{_pnl_color(risk.avg_pnl)}]${risk.avg_pnl:+.2f}[/]",
        "",
        f"[bold]Open Positions:[/] {len(risk.positions)}  |  "
        f"[bold]Scans:[/] {scan_count}  |  "
        f"[bold]Uptime:[/] {uptime_seconds/3600:.1f}h",
    ]

    if risk.consecutive_losses > 0:
        lines.append(f"\n[yellow]Consecutive losses: {risk.consecutive_losses}[/]")
    if risk.heat_level in ("ORANGE", "RED"):
        lines.append(f"\n[{color}]HEAT WARNING: Sizing reduced, min edge raised[/]")

    return Panel(
        "\n".join(lines),
        title="[bold cyan]ValueBetting Status[/]",
        border_style="cyan",
        box=box.ROUNDED,
    )


def build_positions_table(positions: dict[str, Position]) -> Panel:
    """Build the open positions table."""
    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta")
    table.add_column("Market", max_width=40)
    table.add_column("Cat", width=8)
    table.add_column("Side", width=5)
    table.add_column("Entry", width=7)
    table.add_column("Size", width=8)
    table.add_column("TrueP", width=7)
    table.add_column("Age", width=8)

    if not positions:
        table.add_row("[dim]No open positions[/]", "", "", "", "", "", "")
    else:
        for cid, pos in positions.items():
            age_min = pos.age_seconds / 60
            age_str = f"{age_min:.0f}m" if age_min < 60 else f"{age_min/60:.1f}h"

            table.add_row(
                pos.market_question[:38],
                f"[bold]{pos.category}[/]",
                f"[cyan]{pos.side}[/]",
                f"${pos.entry_price:.3f}",
                f"${pos.size_usd:.2f}",
                f"{pos.true_prob_at_entry:.2f}",
                age_str,
            )

    return Panel(table, title="[bold magenta]Open Positions[/]", border_style="magenta")


def build_opportunities_table(signals: list[EdgeSignal]) -> Panel:
    """Build the latest opportunities table."""
    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold green")
    table.add_column("Market", max_width=40)
    table.add_column("Cat", width=8)
    table.add_column("Side", width=5)
    table.add_column("Edge", width=7)
    table.add_column("TrueP", width=7)
    table.add_column("Price", width=7)
    table.add_column("Sources", width=8)
    table.add_column("Action", width=8)

    actionable = [s for s in signals if s.is_actionable][:8]
    if not actionable:
        table.add_row("[dim]No opportunities found[/]", "", "", "", "", "", "", "")
    else:
        for sig in actionable:
            edge_color = "bold green" if sig.net_edge >= 0.08 else "green"
            table.add_row(
                sig.market.question[:38],
                f"[bold]{sig.market.category}[/]",
                f"[cyan]{sig.side}[/]",
                f"[{edge_color}]{sig.net_edge:.1%}[/]",
                f"{sig.true_prob:.3f}",
                f"${sig.entry_price:.3f}",
                f"{sig.consensus.sources}",
                f"[bold green]BUY[/]" if sig.is_actionable else "[dim]SKIP[/]",
            )

    return Panel(table, title="[bold green]Latest Opportunities[/]", border_style="green")


def build_recent_trades_table(trades: list[dict]) -> Panel:
    """Build recent trades table from logger stats."""
    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold yellow")
    table.add_column("Market", max_width=30)
    table.add_column("Cat", width=7)
    table.add_column("P&L", width=8)
    table.add_column("Edge", width=7)
    table.add_column("Exit", width=14)

    if not trades:
        table.add_row("[dim]No trades yet[/]", "", "", "", "")
    else:
        for t in trades[:8]:
            pnl = t.get("pnl_usd", 0)
            table.add_row(
                str(t.get("market_question", ""))[:28],
                str(t.get("category", "")),
                f"[{_pnl_color(pnl)}]${pnl:+.2f}[/]",
                f"{t.get('net_edge_at_entry', 0):.1%}",
                str(t.get("exit_reason", ""))[:12],
            )

    return Panel(table, title="[bold yellow]Recent Trades[/]", border_style="yellow")


def build_ml_panel(calibration_model=None, edge_model=None, brier_scores: dict = None) -> Panel:
    """Build ML model status panel."""
    lines = []

    # Calibration model status
    cal_status = "[green]ACTIVE[/]" if calibration_model and calibration_model.is_trained else "[dim]Collecting data...[/]"
    lines.append(f"[bold]Probability Calibration:[/] {cal_status}")

    # Edge decay model status
    edge_status = "[green]ACTIVE[/]" if edge_model and edge_model.is_trained else "[dim]Collecting data...[/]"
    lines.append(f"[bold]Edge Decay Predictor:[/]  {edge_status}")

    # Brier scores per source
    if brier_scores:
        lines.append("")
        lines.append("[bold]Source Calibration (Brier, lower=better):[/]")
        for source, score in sorted(brier_scores.items(), key=lambda x: x[1]):
            quality = "green" if score < 0.15 else "yellow" if score < 0.25 else "red"
            bar_len = int((1 - score) * 20)
            bar = "[" + "#" * bar_len + "." * (20 - bar_len) + "]"
            lines.append(f"  {source:<16} [{quality}]{score:.3f}[/] {bar}")

    return Panel(
        "\n".join(lines) if lines else "[dim]ML models initializing...[/]",
        title="[bold blue]Machine Learning[/]",
        border_style="blue",
    )


def build_category_panel(stats: dict) -> Panel:
    """Build per-category performance panel."""
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Category", width=10)
    table.add_column("Trades", width=7)
    table.add_column("Win%", width=7)
    table.add_column("P&L", width=9)
    table.add_column("Avg Edge", width=9)

    by_cat = stats.get("by_category", [])
    if not by_cat:
        table.add_row("[dim]No data[/]", "", "", "", "")
    else:
        for c in by_cat:
            trades = c.get("trades", 0)
            wins = c.get("wins", 0)
            wr = wins / trades if trades else 0
            pnl = c.get("pnl", 0)
            table.add_row(
                c.get("category", "?"),
                str(trades),
                f"{wr:.0%}",
                f"[{_pnl_color(pnl)}]${pnl:+.2f}[/]",
                f"{c.get('avg_edge', 0):.1%}",
            )

    return Panel(table, title="[bold]Category Performance[/]")


def render_dashboard(
    risk: RiskManager,
    scan_count: int,
    uptime_seconds: float,
    latest_signals: list[EdgeSignal] = None,
    trade_logger: TradeLogger = None,
    calibration_model=None,
    edge_model=None,
    brier_scores: dict = None,
):
    """Render the full dashboard to terminal."""
    console.clear()

    # Header
    console.print(
        "[bold cyan]"
        "╔══════════════════════════════════════════════════════════════════╗\n"
        "║              ValueBetting — Polymarket Value Arb               ║\n"
        "╚══════════════════════════════════════════════════════════════════╝"
        "[/]"
    )
    console.print()

    # Status
    console.print(build_status_panel(risk, scan_count, uptime_seconds))

    # Two columns: Positions + ML
    console.print(build_positions_table(risk.positions))
    console.print(build_ml_panel(calibration_model, edge_model, brier_scores))

    # Opportunities
    if latest_signals:
        console.print(build_opportunities_table(latest_signals))

    # Stats from logger
    if trade_logger:
        stats = trade_logger.get_stats()
        console.print(build_recent_trades_table(stats.get("recent_trades", [])))
        console.print(build_category_panel(stats))

    console.print(f"\n[dim]Last update: {time.strftime('%H:%M:%S')}[/]")
