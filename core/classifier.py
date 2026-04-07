"""
Market Classifier — Assigns categories to Polymarket markets.
Uses keyword matching first, AI fallback for ambiguous cases.
"""

import logging
import re
from typing import Optional

from core.market import Market

logger = logging.getLogger(__name__)

# Keyword rules for fast classification (no API call needed)
CATEGORY_RULES = {
    "sports": {
        "keywords": [
            "nba", "nfl", "mlb", "nhl", "ufc", "mma", "premier league",
            "la liga", "champions league", "mls", "ncaa", "march madness",
            "super bowl", "world series", "stanley cup", "wimbledon",
            "grand slam", "f1", "formula 1", "nascar",
        ],
        "patterns": [
            r"\bvs\.?\b", r"\bbeat\b", r"\bwin\b.*\b(game|match|series|championship)\b",
            r"\b(lakers|celtics|warriors|yankees|dodgers|chiefs|eagles)\b",
        ],
    },
    "weather": {
        "keywords": [
            "temperature", "weather", "rain", "snow", "hurricane", "tornado",
            "heat wave", "cold front", "fahrenheit", "celsius", "precipitation",
            "wind speed", "flood", "drought", "wildfire",
        ],
        "patterns": [
            r"\b\d+\s*°?[fFcC]\b", r"\bhigh\s+temp", r"\blow\s+temp",
            r"\binches\s+of\s+(rain|snow)\b",
        ],
    },
    "politics": {
        "keywords": [
            "election", "president", "senate", "congress", "governor",
            "democrat", "republican", "gop", "vote", "poll", "ballot",
            "primary", "caucus", "electoral", "impeach", "legislation",
            "executive order", "supreme court", "cabinet",
        ],
        "patterns": [
            r"\b(trump|biden|harris|desantis|newsom)\b",
            r"\bwin\b.*\b(state|election|primary|seat)\b",
        ],
    },
    "crypto": {
        "keywords": [
            "bitcoin", "btc", "ethereum", "eth", "crypto", "blockchain",
            "defi", "nft", "solana", "sol", "dogecoin", "doge",
            "altcoin", "stablecoin", "usdc", "usdt", "binance", "coinbase",
            "etf", "halving", "mining",
        ],
        "patterns": [
            r"\$\d+[kK]?\s*(btc|eth|bitcoin|ethereum)",
            r"\b(btc|eth|sol)\b.*\b(price|above|below|reach)\b",
        ],
    },
    "economics": {
        "keywords": [
            "gdp", "inflation", "cpi", "fed", "interest rate", "unemployment",
            "recession", "stock market", "s&p", "nasdaq", "dow jones",
            "treasury", "bond", "yield", "tariff", "trade deficit",
            "federal reserve", "fomc",
        ],
        "patterns": [
            r"\b(fed|fomc)\b.*\b(rate|cut|hike)\b",
        ],
    },
}


def classify_market(market: Market) -> str:
    """Classify a market by category using keyword + pattern matching.

    Returns category string: sports/weather/politics/crypto/economics/other
    """
    question = market.question.lower()
    raw_category = market.category.lower() if market.category else ""
    tags = " ".join(str(t).lower() for t in market.raw.get("tags", []))
    text = f"{question} {raw_category} {tags}"

    best_category = "other"
    best_score = 0

    for category, rules in CATEGORY_RULES.items():
        score = 0

        # Keyword hits
        for kw in rules["keywords"]:
            if kw in text:
                score += 2

        # Pattern hits
        for pattern in rules["patterns"]:
            if re.search(pattern, text, re.IGNORECASE):
                score += 3

        if score > best_score:
            best_score = score
            best_category = category

    if best_score >= 2:
        logger.debug(f"Classified '{question[:50]}' -> {best_category} (score={best_score})")
        return best_category

    # Low confidence — return "other" (AI fallback can be called separately)
    logger.debug(f"Low-confidence classification for '{question[:50]}' -> other")
    return "other"
