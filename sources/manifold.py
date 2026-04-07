"""
Manifold Markets API — Cross-reference odds source.
Public API, no authentication required.
"""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.manifold.markets/v0"


async def search_markets(query: str, limit: int = 5) -> list[dict]:
    """Search Manifold Markets by query string."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{BASE_URL}/search-markets",
                params={"term": query, "limit": limit, "filter": "open"},
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as e:
        logger.error(f"Manifold search failed: {e}")
        return []


async def get_market_probability(market_id: str) -> Optional[float]:
    """Get current probability for a Manifold market."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{BASE_URL}/market/{market_id}")
            resp.raise_for_status()
            data = resp.json()
            return data.get("probability")
    except httpx.HTTPError as e:
        logger.error(f"Manifold market fetch failed: {e}")
        return None


async def find_matching_probability(question: str) -> Optional[tuple[float, str]]:
    """Search Manifold for a matching market and return its probability.

    Returns (probability, market_url) or None if no match found.
    """
    # Extract key terms from the question for searching
    results = await search_markets(question, limit=3)

    if not results:
        return None

    # Take the top result — Manifold's search is relevance-ranked
    best = results[0]
    prob = best.get("probability")
    url = best.get("url", "")

    if prob is not None:
        logger.info(f"Manifold match: '{best.get('question', '')[:60]}' -> {prob:.3f}")
        return (prob, url)

    return None
