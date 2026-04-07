"""
Exit Engine — Determines when to close open positions.

Exit conditions:
1. Edge reversion (take profit) — market corrected toward fair value
2. Edge gone — no more mispricing
3. Stop loss — price moved against us
4. Time stop — held too long without profit
5. Consensus flip — our estimate crossed to other side of market price
"""

import time
import logging

from config import settings
from core.market import Position, ConsensusEstimate

logger = logging.getLogger(__name__)


def should_exit(
    position: Position,
    current_price: float,
    current_consensus: ConsensusEstimate = None,
    ml_exit_model=None,
) -> tuple[bool, str]:
    """Determine if an open position should be exited.

    Args:
        position: The open position
        current_price: Current market price for the position's side
        current_consensus: Updated consensus estimate (if available)
        ml_exit_model: Optional ML model for exit timing prediction

    Returns:
        (should_exit: bool, reason: str)
    """
    entry_price = position.entry_price
    true_prob = position.true_prob_at_entry

    # Use updated consensus if available
    if current_consensus and current_consensus.confidence > 0:
        true_prob = current_consensus.probability
        if position.side == "NO":
            true_prob = 1.0 - true_prob

    # Current edge and P&L
    if position.side == "YES":
        current_edge = true_prob - current_price
    else:
        current_edge = (1.0 - true_prob) - current_price

    pnl = current_price - entry_price

    # ── ML Exit Model (if available) ──
    if ml_exit_model is not None:
        try:
            should, confidence, ml_reason = ml_exit_model.predict(
                edge=current_edge,
                pnl=pnl,
                age_seconds=position.age_seconds,
                category=position.category,
                entry_price=entry_price,
                current_price=current_price,
            )
            if should and confidence > 0.7:
                return True, f"ML exit: {ml_reason} (conf={confidence:.2f})"
        except Exception as e:
            logger.debug(f"ML exit model failed: {e}")

    # ── Rule-Based Exits ──

    # 1. Price corrected toward fair value — take profit
    if current_edge < settings.EXIT_EDGE_PCT and pnl > 0:
        return True, f"Take profit: edge={current_edge:.1%}, P&L={pnl:+.3f}"

    # 2. Edge gone — no more mispricing
    if current_edge <= 0:
        return True, f"Edge gone: edge={current_edge:.1%}, P&L={pnl:+.3f}"

    # 3. Stop loss
    if pnl < -settings.STOP_LOSS_PER_CONTRACT:
        return True, f"Stop loss: P&L={pnl:+.3f}"

    # 4. Time stop — held too long without profit
    if position.age_seconds > settings.MAX_HOLD_SECONDS and pnl <= 0:
        return True, f"Time stop: held {position.age_seconds/3600:.1f}h, P&L={pnl:+.3f}"

    # 5. Consensus flip — our estimate moved to other side of market
    if current_consensus and current_consensus.confidence > 0.5:
        consensus_prob = current_consensus.probability
        if position.side == "YES" and consensus_prob < current_price:
            return True, f"Consensus flip: estimate {consensus_prob:.3f} < market {current_price:.3f}"
        elif position.side == "NO" and (1 - consensus_prob) < current_price:
            return True, f"Consensus flip: NO estimate {1-consensus_prob:.3f} < market {current_price:.3f}"

    return False, ""
