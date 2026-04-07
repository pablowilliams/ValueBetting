"""
Crypto Estimator — Uses free CoinGecko data + sentiment.
No API key required for basic functionality.
"""

import re
import math
import logging
from typing import Optional

import httpx

from config import settings
from core.market import Market, ProbEstimate
from estimators.base import Estimator

logger = logging.getLogger(__name__)

# Coin name -> CoinGecko ID
COIN_IDS = {
    "bitcoin": "bitcoin", "btc": "bitcoin",
    "ethereum": "ethereum", "eth": "ethereum",
    "solana": "solana", "sol": "solana",
    "dogecoin": "dogecoin", "doge": "dogecoin",
    "bnb": "binancecoin", "binance": "binancecoin",
    "xrp": "ripple", "ripple": "ripple",
    "cardano": "cardano", "ada": "cardano",
    "polkadot": "polkadot", "dot": "polkadot",
    "avalanche": "avalanche-2", "avax": "avalanche-2",
    "chainlink": "chainlink", "link": "chainlink",
}


def _extract_coin(question: str) -> Optional[str]:
    """Extract coin ID from a market question."""
    q = question.lower()
    for name, coin_id in COIN_IDS.items():
        if name in q:
            return coin_id
    return None


def _extract_price_threshold(question: str) -> Optional[tuple[float, str]]:
    """Extract price threshold and direction from question.

    Returns (price, direction) where direction is 'above', 'below', or 'between'.
    """
    q = question.lower()

    # "$100,000" or "$100000" or "$100K"
    def parse_price(s: str) -> float:
        s = s.replace(",", "").replace("$", "").strip()
        if s.lower().endswith("k"):
            return float(s[:-1]) * 1000
        if s.lower().endswith("m"):
            return float(s[:-1]) * 1_000_000
        return float(s)

    # "above $X" / "hit $X" / "reach $X"
    m = re.search(r'(?:above|hit|reach|exceed|over)\s+\$?([\d,]+[kKmM]?)', q)
    if m:
        return (parse_price(m.group(1)), "above")

    # "below $X" / "under $X"
    m = re.search(r'(?:below|under|drop to)\s+\$?([\d,]+[kKmM]?)', q)
    if m:
        return (parse_price(m.group(1)), "below")

    # "between $X and $Y"
    m = re.search(r'between\s+\$?([\d,]+[kKmM]?)\s+and\s+\$?([\d,]+[kKmM]?)', q)
    if m:
        low = parse_price(m.group(1))
        high = parse_price(m.group(2))
        return ((low + high) / 2, "between")

    # "price of X be between $A and $B"
    m = re.search(r'\$?([\d,]+[kKmM]?)\s*(?:and|to|-)\s*\$?([\d,]+[kKmM]?)', q)
    if m:
        low = parse_price(m.group(1))
        high = parse_price(m.group(2))
        if low > 0 and high > low:
            return ((low + high) / 2, "between")

    return None


async def get_coin_price(coin_id: str) -> Optional[dict]:
    """Fetch current price and volatility from CoinGecko (free, no key)."""
    try:
        headers = {}
        if settings.COINGECKO_API_KEY:
            headers["x-cg-demo-api-key"] = settings.COINGECKO_API_KEY

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"https://api.coingecko.com/api/v3/coins/{coin_id}",
                params={"localization": "false", "tickers": "false",
                        "community_data": "false", "developer_data": "false"},
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            market = data.get("market_data", {})

            return {
                "price": market.get("current_price", {}).get("usd", 0),
                "change_24h": market.get("price_change_percentage_24h", 0) or 0,
                "change_7d": market.get("price_change_percentage_7d", 0) or 0,
                "change_30d": market.get("price_change_percentage_30d", 0) or 0,
                "ath": market.get("ath", {}).get("usd", 0),
                "atl": market.get("atl", {}).get("usd", 0),
            }
    except httpx.HTTPError as e:
        logger.debug(f"CoinGecko fetch failed for {coin_id}: {e}")
        return None


class CryptoEstimator(Estimator):
    """Estimates probability for crypto price markets."""

    @property
    def source_name(self) -> str:
        return "crypto_model"

    async def estimate(self, market: Market) -> ProbEstimate | None:
        if market.category != "crypto":
            return None

        coin_id = _extract_coin(market.question)
        if not coin_id:
            return None

        threshold = _extract_price_threshold(market.question)
        if not threshold:
            return None

        target_price, direction = threshold

        coin_data = await get_coin_price(coin_id)
        if not coin_data or coin_data["price"] <= 0:
            return None

        current_price = coin_data["price"]

        # Estimate daily volatility from recent price changes
        daily_vol = abs(coin_data["change_24h"]) / 100
        if daily_vol < 0.01:
            daily_vol = 0.03  # Default 3% daily vol for crypto

        # How many standard deviations away is the target?
        # Assume ~7 day horizon for most markets
        weekly_vol = daily_vol * math.sqrt(7)
        if weekly_vol < 0.01:
            weekly_vol = 0.08

        price_ratio = target_price / current_price
        z_score = (price_ratio - 1.0) / weekly_vol

        # Normal CDF approximation
        prob_above = 1.0 / (1.0 + math.exp(1.7 * z_score))

        if direction == "above":
            prob = prob_above
        elif direction == "below":
            prob = 1.0 - prob_above
        else:  # between
            # Narrow range — lower probability
            prob = min(0.35, prob_above * 0.5)

        prob = max(0.02, min(0.98, prob))
        confidence = 0.55  # Crypto is volatile, moderate confidence

        reasoning = (
            f"CoinGecko: {coin_id} current=${current_price:,.0f}, "
            f"target=${target_price:,.0f} ({direction}), "
            f"24h={coin_data['change_24h']:+.1f}%, vol={weekly_vol:.1%}"
        )
        logger.info(f"[CRYPTO] {reasoning} -> prob={prob:.3f}")

        return ProbEstimate(
            probability=prob,
            confidence=confidence,
            source=self.source_name,
            source_detail="coingecko",
            reasoning=reasoning,
        )
