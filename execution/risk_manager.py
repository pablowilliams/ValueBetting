"""
Risk Manager — Drawdown Heat System + Circuit Breakers
"""

import time
import json
import logging
from dataclasses import asdict
from typing import Optional

from config import settings
from core.market import Position, TradeRecord

logger = logging.getLogger(__name__)


class RiskManager:
    """Manages risk across the trading session."""

    def __init__(self, initial_bankroll: float = None):
        if initial_bankroll is None:
            initial_bankroll = settings.INITIAL_BANKROLL
        self.bankroll = initial_bankroll
        self.initial_bankroll = initial_bankroll
        self.high_water_mark = initial_bankroll
        self.daily_pnl = 0.0
        self.trade_count = 0
        self.consecutive_losses = 0
        self.pause_until = 0.0
        self.session_start = time.time()

        self.positions: dict[str, Position] = {}
        self.trade_history: list[TradeRecord] = []

    # ── Heat System ────────────────────────────────────────────

    @property
    def drawdown(self) -> float:
        if self.high_water_mark <= 0:
            return 0.0
        return (self.high_water_mark - self.bankroll) / self.high_water_mark

    @property
    def heat_level(self) -> str:
        dd = self.drawdown
        if dd >= settings.HEAT_RED:
            return "RED"
        elif dd >= settings.HEAT_ORANGE:
            return "ORANGE"
        elif dd >= settings.HEAT_YELLOW:
            return "YELLOW"
        return "GREEN"

    @property
    def sizing_multiplier(self) -> float:
        level = self.heat_level
        if level == "RED":
            return 0.0
        elif level == "ORANGE":
            return 0.25
        elif level == "YELLOW":
            return 0.50
        return 1.0

    @property
    def min_edge_adjustment(self) -> float:
        level = self.heat_level
        if level == "ORANGE":
            return 0.04
        elif level == "YELLOW":
            return 0.02
        return 0.0

    @property
    def effective_min_edge(self) -> float:
        return settings.MIN_EDGE_PCT + self.min_edge_adjustment

    def category_exposure(self, category: str) -> float:
        """Total USD exposure in a given category."""
        return sum(
            p.size_usd for p in self.positions.values()
            if p.category == category
        )

    # ── Pre-Trade Checks ──────────────────────────────────────

    def can_trade(self, category: str = "") -> tuple[bool, str]:
        if self.heat_level == "RED":
            return False, f"HEAT RED: drawdown {self.drawdown:.1%} — halted"

        if self.bankroll < self.high_water_mark * settings.BALANCE_FLOOR_PCT:
            return False, f"Below balance floor: ${self.bankroll:.2f}"

        if self.daily_pnl < 0 and abs(self.daily_pnl) >= self.initial_bankroll * settings.DAILY_LOSS_LIMIT_PCT:
            return False, f"Daily loss limit: ${self.daily_pnl:.2f}"

        if self.trade_count >= settings.MAX_TRADES_PER_SESSION:
            return False, f"Session trade limit: {self.trade_count}/{settings.MAX_TRADES_PER_SESSION}"

        if time.time() < self.pause_until:
            remaining = self.pause_until - time.time()
            return False, f"Consecutive loss pause: {remaining:.0f}s remaining"

        if len(self.positions) >= settings.MAX_CONCURRENT_POSITIONS:
            return False, f"Max positions: {len(self.positions)}/{settings.MAX_CONCURRENT_POSITIONS}"

        if category:
            cat_exposure = self.category_exposure(category)
            max_cat = self.bankroll * settings.MAX_CATEGORY_EXPOSURE_PCT
            if cat_exposure >= max_cat:
                return False, f"Category '{category}' exposure ${cat_exposure:.2f} >= ${max_cat:.2f}"

        return True, "OK"

    # ── Position Management ───────────────────────────────────

    def open_position(self, position: Position):
        self.positions[position.condition_id] = position
        self.trade_count += 1
        logger.info(
            f"OPENED: {position.market_question[:50]} | {position.side} @ {position.entry_price:.3f} | "
            f"${position.size_usd:.2f} | [{position.category}] | Open: {len(self.positions)}"
        )

    def close_position(
        self,
        condition_id: str,
        exit_price: float,
        exit_reason: str,
    ) -> Optional[TradeRecord]:
        position = self.positions.pop(condition_id, None)
        if not position:
            logger.warning(f"No position found for {condition_id}")
            return None

        pnl_per_contract = exit_price - position.entry_price
        pnl_usd = pnl_per_contract * position.num_contracts
        pnl_pct = pnl_usd / position.size_usd if position.size_usd > 0 else 0

        self.bankroll += pnl_usd
        self.daily_pnl += pnl_usd

        if self.bankroll > self.high_water_mark:
            self.high_water_mark = self.bankroll

        if pnl_usd < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= settings.CONSECUTIVE_LOSS_PAUSE:
                self.pause_until = time.time() + settings.PAUSE_DURATION_SECONDS
                logger.warning(
                    f"CIRCUIT BREAKER: {self.consecutive_losses} consecutive losses — "
                    f"pausing {settings.PAUSE_DURATION_SECONDS}s"
                )
        else:
            self.consecutive_losses = 0

        record = TradeRecord(
            condition_id=condition_id,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            size_usd=position.size_usd,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            true_prob=position.true_prob_at_entry,
            market_price_at_entry=position.entry_price,
            net_edge_at_entry=position.true_prob_at_entry - position.entry_price,
            category=position.category,
            sources_used=position.consensus_at_entry.source_names,
            entry_time=position.entry_time,
            exit_time=time.time(),
            exit_reason=exit_reason,
            market_question=position.market_question,
        )
        self.trade_history.append(record)
        self._log_trade(record)

        logger.info(
            f"CLOSED: {position.market_question[:50]} | {exit_reason} | "
            f"P&L: ${pnl_usd:+.2f} ({pnl_pct:+.1%}) | "
            f"Bank: ${self.bankroll:.2f} | DD: {self.drawdown:.1%} [{self.heat_level}]"
        )
        return record

    def _log_trade(self, record: TradeRecord):
        try:
            with open(settings.TRADE_LOG_FILE, "a") as f:
                f.write(json.dumps(asdict(record)) + "\n")
        except IOError as e:
            logger.error(f"Failed to log trade: {e}")

    # ── Stats ─────────────────────────────────────────────────

    @property
    def win_rate(self) -> float:
        if not self.trade_history:
            return 0.0
        wins = sum(1 for t in self.trade_history if t.pnl_usd > 0)
        return wins / len(self.trade_history)

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl_usd for t in self.trade_history)

    @property
    def avg_pnl(self) -> float:
        if not self.trade_history:
            return 0.0
        return self.total_pnl / len(self.trade_history)

    def status_summary(self) -> str:
        return (
            f"Bank: ${self.bankroll:.2f} | "
            f"P&L: ${self.daily_pnl:+.2f} | "
            f"DD: {self.drawdown:.1%} [{self.heat_level}] | "
            f"Trades: {self.trade_count}/{settings.MAX_TRADES_PER_SESSION} | "
            f"Open: {len(self.positions)} | "
            f"WR: {self.win_rate:.0%} ({len(self.trade_history)} trades) | "
            f"Consec L: {self.consecutive_losses}"
        )
