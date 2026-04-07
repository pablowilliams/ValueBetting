"""
Sentiment Aggregator — Combines free APIs into a -1 to +1 score.
Adapted from weather-bot. Generalized beyond crypto.
"""

import os
import time
import logging
from dataclasses import dataclass
from typing import Optional

import httpx

from config import settings

logger = logging.getLogger(__name__)

_cache: dict = {}
_CACHE_TTL = 300.0


def _get_cached(key: str) -> Optional[dict]:
    entry = _cache.get(key)
    if entry and (time.time() - entry["ts"]) < _CACHE_TTL:
        return entry["data"]
    return None


def _set_cached(key: str, data: dict):
    _cache[key] = {"data": data, "ts": time.time()}


@dataclass
class SentimentSnapshot:
    fear_greed: float = 0.0
    news_sentiment: float = 0.0
    social_sentiment: float = 0.0
    market_momentum: float = 0.0
    composite: float = 0.0
    sources_available: int = 0
    timestamp: float = 0.0

    @property
    def label(self) -> str:
        if self.composite > 0.5:
            return "extreme_greed"
        elif self.composite > 0.2:
            return "greed"
        elif self.composite > -0.2:
            return "neutral"
        elif self.composite > -0.5:
            return "fear"
        else:
            return "extreme_fear"


async def fetch_fear_greed() -> float:
    cached = _get_cached("fear_greed")
    if cached is not None:
        return cached.get("score", 0.0)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get("https://api.alternative.me/fng/?limit=1")
            resp.raise_for_status()
            data = resp.json()
            value = int(data["data"][0]["value"])
            score = (value - 50) / 50.0
            _set_cached("fear_greed", {"score": score})
            logger.info(f"[SENTIMENT] Fear & Greed: {value}/100 -> {score:+.2f}")
            return score
    except Exception as e:
        logger.debug(f"Fear & Greed fetch failed: {e}")
        return 0.0


async def fetch_newsapi_sentiment(topic: str = "bitcoin OR btc") -> float:
    api_key = settings.NEWSAPI_KEY
    if not api_key:
        return 0.0
    cached = _get_cached(f"newsapi_{topic}")
    if cached is not None:
        return cached.get("score", 0.0)

    BULLISH = {"surge", "soar", "rally", "bull", "breakout", "high", "record",
               "adoption", "approved", "gains", "pump", "buy", "bullish", "upgrade"}
    BEARISH = {"crash", "plunge", "dump", "bear", "sell", "ban", "hack",
               "fraud", "bubble", "collapse", "regulation", "crackdown", "fear",
               "bearish", "liquidation", "scam"}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://newsapi.org/v2/everything",
                params={"q": topic, "language": "en", "sortBy": "publishedAt",
                        "pageSize": 20, "apiKey": api_key},
            )
            resp.raise_for_status()
            articles = resp.json().get("articles", [])

        bull = bear = 0
        for a in articles:
            text = f"{(a.get('title') or '').lower()} {(a.get('description') or '').lower()}"
            bull += sum(1 for w in BULLISH if w in text)
            bear += sum(1 for w in BEARISH if w in text)

        total = bull + bear
        score = (bull - bear) / total if total else 0.0
        score = max(-1.0, min(1.0, score))
        _set_cached(f"newsapi_{topic}", {"score": score})
        logger.info(f"[SENTIMENT] NewsAPI ({topic}): bull={bull} bear={bear} -> {score:+.2f}")
        return score
    except Exception as e:
        logger.debug(f"NewsAPI fetch failed: {e}")
        return 0.0


async def fetch_coingecko_momentum() -> float:
    cached = _get_cached("coingecko")
    if cached is not None:
        return cached.get("score", 0.0)
    try:
        headers = {}
        if settings.COINGECKO_API_KEY:
            headers["x-cg-demo-api-key"] = settings.COINGECKO_API_KEY

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://api.coingecko.com/api/v3/coins/bitcoin",
                params={"localization": "false", "tickers": "false",
                        "community_data": "false", "developer_data": "false"},
                headers=headers,
            )
            resp.raise_for_status()
            market = resp.json().get("market_data", {})

        p24 = market.get("price_change_percentage_24h", 0) or 0
        p7d = market.get("price_change_percentage_7d", 0) or 0
        daily = max(-1.0, min(1.0, p24 / 5.0))
        weekly = max(-1.0, min(1.0, p7d / 10.0))
        score = max(-1.0, min(1.0, daily * 0.7 + weekly * 0.3))
        _set_cached("coingecko", {"score": score})
        logger.info(f"[SENTIMENT] CoinGecko: 24h={p24:+.1f}% 7d={p7d:+.1f}% -> {score:+.2f}")
        return score
    except Exception as e:
        logger.debug(f"CoinGecko fetch failed: {e}")
        return 0.0


async def get_sentiment_snapshot(topic: str = "bitcoin OR btc") -> SentimentSnapshot:
    """Fetch all sentiment sources and combine."""
    import asyncio
    results = await asyncio.gather(
        fetch_fear_greed(),
        fetch_newsapi_sentiment(topic),
        fetch_coingecko_momentum(),
        return_exceptions=True,
    )

    def safe(val) -> float:
        return val if isinstance(val, (int, float)) else 0.0

    fg = safe(results[0])
    news = safe(results[1])
    cg = safe(results[2])

    available = sum(1 for s in [fg, news, cg] if s != 0.0)

    weights, scores = [], []
    if fg != 0.0:
        weights.append(0.30); scores.append(fg)
    if news != 0.0:
        weights.append(0.30); scores.append(news)
    if cg != 0.0:
        weights.append(0.40); scores.append(cg)

    composite = sum(w * s for w, s in zip(weights, scores)) / sum(weights) if weights else 0.0

    return SentimentSnapshot(
        fear_greed=fg, news_sentiment=news, market_momentum=cg,
        composite=composite, sources_available=available, timestamp=time.time(),
    )
