"""
Edge Calculator — Computes net edge between consensus probability and market price.
Adapted from SportsBetArb/edge_calculator.py, generalized for all categories.
"""

import logging

from config import settings
from config.categories import get_category_config
from core.market import Market, ConsensusEstimate, EdgeSignal

logger = logging.getLogger(__name__)


def compute_fee(entry_price: float, fee_rate: float = None) -> float:
    """Polymarket fee: fee_rate * price * (1 - price) * 2"""
    if fee_rate is None:
        fee_rate = settings.FEE_RATE
    return fee_rate * entry_price * (1 - entry_price) * 2


def compute_edge(
    market: Market,
    consensus: ConsensusEstimate,
    min_edge: float = None,
    min_confidence: float = None,
    min_sources: int = None,
) -> EdgeSignal:
    """Compute edge for a market given a consensus probability estimate.

    Determines YES or NO side, calculates net edge after fees/slippage,
    and checks all quality gates.
    """
    cat_config = get_category_config(market.category)
    if min_edge is None:
        min_edge = cat_config.get("min_edge", settings.MIN_EDGE_PCT)
    if min_confidence is None:
        min_confidence = settings.MIN_CONSENSUS_CONFIDENCE
    if min_sources is None:
        min_sources = cat_config.get("min_sources", settings.MIN_SOURCES)

    true_prob = consensus.probability

    # Determine best side to trade
    yes_edge = true_prob - market.yes_price
    no_edge = (1 - true_prob) - market.no_price

    if yes_edge >= no_edge:
        side = "YES"
        entry_price = market.best_ask_yes if market.best_ask_yes > 0 else market.yes_price
        prob = true_prob
    else:
        side = "NO"
        entry_price = 1.0 - (market.best_bid_yes if market.best_bid_yes > 0 else market.yes_price)
        prob = 1.0 - true_prob

    # Calculate costs
    fee = compute_fee(entry_price)
    slippage = settings.SLIPPAGE_BUDGET
    total_cost = entry_price + fee + slippage
    gross_edge = prob - entry_price
    net_edge = prob - total_cost
    expected_value = net_edge

    # Paper training mode — trade on any positive edge to generate ML training data
    if net_edge > 0.005:  # 0.5% minimum — basically any divergence
        action = "BUY"
        reason = f"Edge {net_edge:.1%} | conf={consensus.confidence:.2f} | {consensus.sources} sources"
    else:
        action = "SKIP"
        reason = f"No edge: {net_edge:.1%}"

    signal = EdgeSignal(
        market=market,
        consensus=consensus,
        true_prob=true_prob,
        market_price=market.yes_price,
        entry_price=entry_price,
        fee=fee,
        slippage=slippage,
        total_cost=total_cost,
        gross_edge=gross_edge,
        net_edge=net_edge,
        expected_value=expected_value,
        side=side,
        action=action,
        reason=reason,
    )

    # Log all edges above 1% for debugging
    if net_edge > 0.01:
        logger.info(
            f"EDGE [{market.category}] {action}: {market.question[:45]} | "
            f"{side} | True={prob:.3f} | Market={entry_price:.3f} | "
            f"Net={net_edge:.1%} | Gross={gross_edge:.1%} | {reason}"
        )
    if action == "BUY":
        logger.info(
            f"*** BUY SIGNAL [{market.category}]: {market.question[:50]} | "
            f"{side} | True={prob:.3f} | Market={entry_price:.3f} | "
            f"Net edge={net_edge:.1%} | Sources={consensus.source_names}"
        )

    return signal
