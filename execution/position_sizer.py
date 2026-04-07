"""
Position Sizer — Corrected Kelly for Binary Contracts
"""

import logging
from config import settings

logger = logging.getLogger(__name__)


def corrected_kelly(
    true_prob: float,
    entry_price: float,
    fee_rate: float = None,
) -> float:
    """Compute corrected Kelly fraction for binary contracts."""
    if fee_rate is None:
        fee_rate = settings.FEE_RATE

    fee = fee_rate * entry_price * (1 - entry_price) * 2
    win_amount = 1.0 - entry_price - fee
    lose_amount = entry_price + fee

    if win_amount <= 0:
        return 0.0

    kelly_full = (true_prob * win_amount - (1 - true_prob) * lose_amount) / win_amount
    if kelly_full <= 0:
        return 0.0

    return kelly_full


def compute_position_size(
    bankroll: float,
    true_prob: float,
    entry_price: float,
    sizing_mult: float = 1.0,
) -> float:
    """Compute dollar position size with Kelly + heat adjustment."""
    kelly = corrected_kelly(true_prob, entry_price)
    if kelly <= 0:
        return 0.0

    kelly_quarter = kelly * settings.KELLY_FRACTION
    kelly_adjusted = kelly_quarter * sizing_mult
    raw_size = bankroll * kelly_adjusted
    max_bet = bankroll * settings.MAX_BET_PCT
    position_size = min(raw_size, max_bet)

    if position_size < settings.MIN_POSITION_USD:
        logger.debug(f"Position ${position_size:.2f} below minimum ${settings.MIN_POSITION_USD}")
        return 0.0

    num_contracts = position_size / entry_price
    logger.info(
        f"Position sizing: kelly_full={kelly:.3f} | quarter={kelly_quarter:.3f} | "
        f"heat_adj={kelly_adjusted:.3f} | size=${position_size:.2f} | "
        f"contracts={num_contracts:.1f} | entry={entry_price:.3f}"
    )
    return position_size
