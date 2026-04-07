#!/usr/bin/env python3
"""
ValueBetting — Main Bot Orchestrator
--------------------------------------
Multi-category value betting engine for Polymarket.

Core loop:
1. SCAN Polymarket markets
2. CLASSIFY by category (sports/weather/politics/crypto/other)
3. ESTIMATE true probability via cross-market + domain data + AI
4. CONSENSUS: weighted aggregation
5. EDGE: buy when mispriced (net edge > threshold)
6. MONITOR: sell when odds revert to fair value

Usage:
    python bot.py                     # Paper trading (default)
    LIVE_MODE=1 python bot.py         # Live trading
"""

import asyncio
import os
import sys
import time
import signal
import logging

from config import settings
from core.classifier import classify_market
from core.consensus import compute_consensus
from core.edge import compute_edge
from core.exit_engine import should_exit
from core.market import Position, EdgeSignal
from estimators.registry import run_estimators
from execution.position_sizer import compute_position_size
from execution.risk_manager import RiskManager
from execution.executor import get_executor
from sources.polymarket import PolymarketScanner
from dashboard.logger import TradeLogger
from dashboard.terminal import render_dashboard
from ml.calibration import CalibrationModel, CalibrationSample
from ml.edge_decay import EdgeDecayModel

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(settings.LOG_FILE),
    ],
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class ValueBettingBot:
    """Main bot orchestrating the full value betting pipeline."""

    def __init__(self):
        self.scanner = PolymarketScanner()
        self.risk = RiskManager()
        self.executor = get_executor()
        self.trade_logger = TradeLogger()
        self.calibration = CalibrationModel()
        self.edge_decay = EdgeDecayModel()

        self.running = True
        self.scan_count = 0
        self.start_time = time.time()
        self.latest_signals: list[EdgeSignal] = []

        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, signum, frame):
        logger.info("Shutting down gracefully...")
        self.running = False

    async def run(self):
        """Main async loop."""
        mode = "LIVE" if settings.LIVE_MODE else "PAPER"
        logger.info(f"{'='*60}")
        logger.info(f"  ValueBetting Bot — {mode} MODE")
        logger.info(f"  Bankroll: ${settings.INITIAL_BANKROLL:.2f}")
        logger.info(f"  Min edge: {settings.MIN_EDGE_PCT:.0%} | Kelly: {settings.KELLY_FRACTION}x")
        logger.info(f"  ML Calibration: {'ACTIVE' if self.calibration.is_trained else 'Collecting data'}")
        logger.info(f"  ML Edge Decay: {'ACTIVE' if self.edge_decay.is_trained else 'Collecting data'}")
        logger.info(f"{'='*60}")

        while self.running:
            try:
                await self._scan_cycle()
                await self._monitor_positions()
                self._render()

                # Periodically retrain ML models
                if self.scan_count % 50 == 0 and self.scan_count > 0:
                    self._retrain_models()

                # Wait for next scan
                for _ in range(settings.SCAN_INTERVAL_SECONDS):
                    if not self.running:
                        break
                    await asyncio.sleep(1)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(30)

        self._final_report()

    async def _scan_cycle(self):
        """One full scan cycle: fetch -> classify -> estimate -> buy."""
        self.scan_count += 1
        logger.info(f"\n{'─'*40} SCAN #{self.scan_count} {'─'*40}")

        can_trade, reason = self.risk.can_trade()
        if not can_trade:
            logger.warning(f"Trading blocked: {reason}")
            return

        # 1. SCAN — Fetch active markets
        markets = await self.scanner.fetch_active_markets(limit=100)
        if not markets:
            logger.warning("No markets fetched")
            return

        # 2. CLASSIFY — Assign categories
        category_counts = {}
        for market in markets:
            market.category = classify_market(market)
            category_counts[market.category] = category_counts.get(market.category, 0) + 1

        logger.info(f"Categories: {category_counts}")

        # 3-5. ESTIMATE + CONSENSUS + EDGE — For each market
        signals = []
        for market in markets:
            # Skip markets we already have positions in
            if market.condition_id in self.risk.positions:
                continue

            # Run estimator pipeline for this market's category
            estimates = await run_estimators(market)
            if not estimates:
                continue

            # Compute consensus (ML model overrides fixed weights if trained)
            ml_model = self.calibration if self.calibration.is_trained else None
            consensus = compute_consensus(estimates, market.category, ml_model=ml_model)

            # Enrich with orderbook data for quality checks
            await self.scanner.enrich_with_orderbook(market)

            # Compute edge with heat-adjusted minimum
            signal = compute_edge(
                market, consensus,
                min_edge=self.risk.effective_min_edge,
            )
            signals.append(signal)

        self.latest_signals = signals

        # Filter and sort by edge
        opportunities = [s for s in signals if s.is_actionable]
        opportunities.sort(key=lambda s: s.net_edge, reverse=True)

        # Log scan
        self.trade_logger.log_scan(
            markets=len(markets),
            opportunities=len(opportunities),
            trades=0,
            categories=category_counts,
        )

        if not opportunities:
            logger.info("No actionable opportunities this scan")
            return

        logger.info(f"Found {len(opportunities)} opportunities:")
        for i, opp in enumerate(opportunities[:5]):
            logger.info(
                f"  #{i+1}: {opp.market.question[:50]} | "
                f"{opp.side} | edge={opp.net_edge:.1%} | "
                f"[{opp.market.category}] | sources={opp.consensus.source_names}"
            )

        # 6. EXECUTE — Place trades
        trades_executed = 0
        for opp in opportunities:
            can_trade, reason = self.risk.can_trade(category=opp.market.category)
            if not can_trade:
                logger.warning(f"Trading blocked: {reason}")
                break

            if opp.market.condition_id in self.risk.positions:
                continue

            trades_executed += self._execute_entry(opp)

    def _execute_entry(self, signal: EdgeSignal) -> int:
        """Execute a single trade entry. Returns 1 if successful, 0 otherwise."""
        market = signal.market
        size_usd = compute_position_size(
            bankroll=self.risk.bankroll,
            true_prob=signal.true_prob if signal.side == "YES" else (1 - signal.true_prob),
            entry_price=signal.entry_price,
            sizing_mult=self.risk.sizing_multiplier,
        )

        if size_usd <= 0:
            return 0

        token_id = market.token_id_yes if signal.side == "YES" else market.token_id_no
        result = self.executor.buy(
            token_id=token_id,
            side=signal.side,
            size_usd=size_usd,
            expected_price=signal.entry_price,
            market_question=market.question,
        )

        if result.success:
            position = Position(
                condition_id=market.condition_id,
                token_id=token_id,
                side=signal.side,
                entry_price=result.fill_price,
                size_usd=size_usd,
                num_contracts=result.filled_size,
                true_prob_at_entry=signal.true_prob,
                consensus_at_entry=signal.consensus,
                category=market.category,
                market_question=market.question,
                entry_time=time.time(),
            )
            self.risk.open_position(position)
            return 1
        else:
            logger.warning(f"Order failed: {result.error}")
            return 0

    async def _monitor_positions(self):
        """Check all open positions for exit conditions."""
        if not self.risk.positions:
            return

        logger.info(f"Monitoring {len(self.risk.positions)} open positions...")
        positions_to_close = []

        for cid, position in self.risk.positions.items():
            market = self.scanner.get_market(cid)
            if not market:
                continue

            await self.scanner.enrich_with_orderbook(market)

            if position.side == "YES":
                current_price = market.best_bid_yes if market.best_bid_yes > 0 else market.yes_price
            else:
                current_price = 1.0 - (market.best_ask_yes if market.best_ask_yes > 0 else market.yes_price)

            # Get updated consensus if possible
            estimates = await run_estimators(market)
            current_consensus = None
            if estimates:
                ml_model = self.calibration if self.calibration.is_trained else None
                current_consensus = compute_consensus(estimates, market.category, ml_model=ml_model)

            # Check exit conditions (ML model used if trained)
            ml_exit = self.edge_decay if self.edge_decay.is_trained else None
            should_sell, reason = should_exit(
                position=position,
                current_price=current_price,
                current_consensus=current_consensus,
                ml_exit_model=ml_exit,
            )

            if should_sell:
                positions_to_close.append((cid, current_price, reason))

            # Log position status
            pnl = (current_price - position.entry_price) * position.num_contracts
            logger.info(
                f"  {position.market_question[:40]} | {position.side} | "
                f"Entry={position.entry_price:.3f} | Now={current_price:.3f} | "
                f"P&L=${pnl:+.2f} | Age={position.age_seconds/60:.0f}m"
            )

        # Execute exits
        for cid, exit_price, reason in positions_to_close:
            position = self.risk.positions.get(cid)
            if not position:
                continue

            result = self.executor.sell(
                token_id=position.token_id,
                side=position.side,
                num_contracts=position.num_contracts,
                expected_price=exit_price,
                market_question=position.market_question,
            )

            if result.success:
                record = self.risk.close_position(cid, result.fill_price, reason)
                if record:
                    self.trade_logger.log_trade(record)

    def _retrain_models(self):
        """Periodically retrain ML models from accumulated data."""
        logger.info("Checking ML model retraining...")
        if self.calibration.train():
            logger.info("Calibration model retrained successfully")
        if self.edge_decay.train():
            logger.info("Edge decay model retrained successfully")

    def _render(self):
        """Render the terminal dashboard."""
        brier = self.calibration.get_source_brier_scores() if self.calibration.is_trained else {}
        render_dashboard(
            risk=self.risk,
            scan_count=self.scan_count,
            uptime_seconds=time.time() - self.start_time,
            latest_signals=self.latest_signals,
            trade_logger=self.trade_logger,
            calibration_model=self.calibration,
            edge_model=self.edge_decay,
            brier_scores=brier,
        )

    def _final_report(self):
        """Print final session report."""
        logger.info(f"\n{'='*60}")
        logger.info("  SESSION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"  Scans: {self.scan_count}")
        logger.info(f"  Trades: {len(self.risk.trade_history)}")
        logger.info(f"  Win rate: {self.risk.win_rate:.1%}")
        logger.info(f"  Total P&L: ${self.risk.total_pnl:+.2f}")
        logger.info(f"  Avg P&L/trade: ${self.risk.avg_pnl:+.2f}")
        logger.info(f"  Final bankroll: ${self.risk.bankroll:.2f}")
        logger.info(f"  Max drawdown: {self.risk.drawdown:.1%}")
        logger.info(f"  Open positions: {len(self.risk.positions)}")

        if self.risk.trade_history:
            winners = [t for t in self.risk.trade_history if t.pnl_usd > 0]
            losers = [t for t in self.risk.trade_history if t.pnl_usd <= 0]
            avg_win = sum(t.pnl_usd for t in winners) / len(winners) if winners else 0
            avg_loss = sum(t.pnl_usd for t in losers) / len(losers) if losers else 0
            logger.info(f"  Avg win: ${avg_win:+.2f} | Avg loss: ${avg_loss:+.2f}")
            if avg_loss != 0:
                logger.info(f"  Win/Loss ratio: {abs(avg_win/avg_loss):.2f}")

        # ML model status
        logger.info(f"\n  ML Calibration: {'Trained' if self.calibration.is_trained else 'Not enough data'}")
        logger.info(f"  ML Edge Decay: {'Trained' if self.edge_decay.is_trained else 'Not enough data'}")

        stats = self.trade_logger.get_stats()
        by_cat = stats.get("by_category", [])
        if by_cat:
            logger.info(f"\n  Category Breakdown:")
            for c in by_cat:
                trades = c.get("trades", 0)
                wins = c.get("wins", 0)
                wr = wins / trades if trades else 0
                logger.info(f"    {c['category']}: {trades} trades, {wr:.0%} WR, ${c.get('pnl', 0):+.2f}")

        logger.info(f"{'='*60}")


def main():
    if os.getenv("LIVE_MODE", "").lower() in ("1", "true", "yes"):
        settings.LIVE_MODE = True

    bot = ValueBettingBot()
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()
