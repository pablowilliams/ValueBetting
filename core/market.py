"""
Core dataclasses for the ValueBetting engine.
All modules share these types for consistent data flow.
"""

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Market:
    """A Polymarket market with pricing and orderbook data."""
    condition_id: str
    question: str
    token_id_yes: str
    token_id_no: str
    yes_price: float
    no_price: float
    volume: float
    liquidity: float
    end_date: str
    category: str              # Assigned by classifier: sports/weather/politics/crypto/other
    spread: float
    best_bid_yes: float
    best_ask_yes: float
    orderbook_depth: float
    timestamp: float
    raw: dict = field(default_factory=dict)

    @property
    def midpoint(self) -> float:
        if self.best_bid_yes and self.best_ask_yes:
            return (self.best_bid_yes + self.best_ask_yes) / 2
        return self.yes_price


@dataclass
class ProbEstimate:
    """A probability estimate from a single source."""
    probability: float         # 0-1: estimated true probability of YES
    confidence: float          # 0-1: how confident this source is
    source: str                # e.g., "cross_market", "sports_odds", "ai_ensemble"
    source_detail: str = ""    # e.g., "metaculus", "draftkings_avg"
    timestamp: float = 0.0
    reasoning: str = ""        # Optional explanation

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class ConsensusEstimate:
    """Weighted aggregation of multiple ProbEstimates."""
    probability: float         # Weighted mean probability
    confidence: float          # Overall confidence (accounts for agreement + source count)
    sources: int               # Number of contributing sources
    estimates: list[ProbEstimate] = field(default_factory=list)

    @property
    def source_names(self) -> list[str]:
        return [e.source for e in self.estimates]


@dataclass
class EdgeSignal:
    """A computed edge signal — actionable if edge is above threshold."""
    market: Market
    consensus: ConsensusEstimate
    true_prob: float           # Consensus probability
    market_price: float        # Current Polymarket price
    entry_price: float         # Actual entry price (best ask)
    fee: float
    slippage: float
    total_cost: float          # entry_price + fee + slippage
    gross_edge: float          # true_prob - market_price
    net_edge: float            # true_prob - total_cost
    expected_value: float
    side: str                  # "YES" or "NO"
    action: str                # "BUY" or "SKIP"
    reason: str

    @property
    def is_actionable(self) -> bool:
        return self.action == "BUY"


@dataclass
class Position:
    """An open position on Polymarket."""
    condition_id: str
    token_id: str
    side: str                  # "YES" or "NO"
    entry_price: float
    size_usd: float
    num_contracts: float
    true_prob_at_entry: float
    consensus_at_entry: ConsensusEstimate
    category: str
    market_question: str
    entry_time: float

    @property
    def age_seconds(self) -> float:
        return time.time() - self.entry_time


@dataclass
class TradeRecord:
    """A completed trade for logging and calibration tracking."""
    condition_id: str
    side: str
    entry_price: float
    exit_price: float
    size_usd: float
    pnl_usd: float
    pnl_pct: float
    true_prob: float
    market_price_at_entry: float
    net_edge_at_entry: float
    category: str
    sources_used: list[str]
    entry_time: float
    exit_time: float
    exit_reason: str
    market_question: str
    actual_outcome: Optional[bool] = None  # Set after resolution for calibration
