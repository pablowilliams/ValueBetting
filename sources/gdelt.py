"""
GDELT Project — Global news event monitoring for probability estimation.
100% free, no API key required. Updates every 15 minutes.

Uses DOC 2.0 API for:
- Volume momentum: rising news coverage = event becoming more likely
- Tone shifts: sentiment changes signal probability changes
- Source diversity: broad independent coverage = high credibility

Rate limit: ~1 request per 5-10 seconds per IP.
"""

import asyncio
import logging
import time
import re
from typing import Optional
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# Rate limiting
_last_request = 0.0
_MIN_INTERVAL = 6.0  # seconds between requests

# Cache (GDELT updates every 15 min, no point re-querying faster)
_cache: dict[str, tuple[float, dict]] = {}
_CACHE_TTL = 900.0  # 15 minutes


# GKG theme codes mapped to Polymarket categories
CATEGORY_THEMES = {
    "politics": ["ELECTION", "LEADER", "LEGISLATION", "POLITICAL_TURMOIL",
                 "IMPEACHMENT", "SCANDAL", "GOV_LEADER"],
    "economics": ["ECON_TRADE", "ECON_INFLATION", "ECON_INTEREST_RATE",
                  "ECON_UNEMPLOYMENT", "ECON_STOCKMARKET", "ECON_GDP",
                  "EPU_POLICY", "EPU_ECONOMY"],
    "crypto": [],  # Use keyword search instead
    "weather": ["CRISISLEX_C01_WEATHER", "NATURAL_DISASTER", "ENV_CLIMATECHANGE"],
    "sports": [],  # Use keyword search instead
    "other": ["MILITARY", "ARMEDCONFLICT", "TERROR", "SANCTION",
              "CEASEFIRE", "PEACE_NEGOTIATION"],
}


@dataclass
class GdeltSignal:
    """Signal extracted from GDELT data."""
    volume_ratio: float      # Current volume / baseline (>2 = breakout)
    tone: float              # Average tone (-10 to +10)
    tone_shift: float        # Recent tone - baseline tone
    article_count: int       # Raw article count in last 24h
    source_count: int        # Number of distinct sources
    momentum: str            # "rising", "stable", "declining"
    probability_adjustment: float  # How much to adjust probability (-0.3 to +0.3)
    reasoning: str


async def _rate_limit():
    global _last_request
    now = time.time()
    elapsed = now - _last_request
    if elapsed < _MIN_INTERVAL:
        await asyncio.sleep(_MIN_INTERVAL - elapsed)
    _last_request = time.time()


def _get_cached(key: str) -> Optional[dict]:
    if key in _cache:
        ts, data = _cache[key]
        if time.time() - ts < _CACHE_TTL:
            return data
    return None


def _set_cached(key: str, data: dict):
    _cache[key] = (time.time(), data)


def _extract_search_terms(question: str) -> str:
    """Extract meaningful search terms from a Polymarket question."""
    # Remove common question words
    q = question.lower()
    remove = ["will", "the", "be", "a", "an", "in", "on", "at", "to", "of",
              "by", "is", "are", "do", "does", "?", "!", ".", ","]
    words = q.split()
    terms = [w for w in words if w not in remove and len(w) > 2]

    # Keep at most 5 most distinctive terms
    return " ".join(terms[:5])


async def fetch_volume_timeline(query: str, timespan: str = "30d") -> Optional[list[dict]]:
    """Fetch article volume timeline from GDELT DOC API."""
    cache_key = f"vol:{query}:{timespan}"
    cached = _get_cached(cache_key)
    if cached:
        return cached

    await _rate_limit()

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(DOC_API, params={
                "query": query,
                "mode": "timelinevolraw",
                "format": "json",
                "timespan": timespan,
            })
            resp.raise_for_status()
            data = resp.json()

        timeline = data.get("timeline", [])
        if timeline and isinstance(timeline[0], dict):
            # Single series
            points = timeline[0].get("data", [])
        elif timeline:
            points = timeline
        else:
            points = []

        _set_cached(cache_key, points)
        return points

    except Exception as e:
        logger.debug(f"GDELT volume fetch failed: {e}")
        return None


async def fetch_tone_timeline(query: str, timespan: str = "30d") -> Optional[list[dict]]:
    """Fetch average tone timeline from GDELT DOC API."""
    cache_key = f"tone:{query}:{timespan}"
    cached = _get_cached(cache_key)
    if cached:
        return cached

    await _rate_limit()

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(DOC_API, params={
                "query": query,
                "mode": "timelinetone",
                "format": "json",
                "timespan": timespan,
            })
            resp.raise_for_status()
            data = resp.json()

        timeline = data.get("timeline", [])
        if timeline and isinstance(timeline[0], dict):
            points = timeline[0].get("data", [])
        elif timeline:
            points = timeline
        else:
            points = []

        _set_cached(cache_key, points)
        return points

    except Exception as e:
        logger.debug(f"GDELT tone fetch failed: {e}")
        return None


async def fetch_article_list(query: str, max_records: int = 25) -> Optional[list[dict]]:
    """Fetch recent articles matching query."""
    cache_key = f"art:{query}:{max_records}"
    cached = _get_cached(cache_key)
    if cached:
        return cached

    await _rate_limit()

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(DOC_API, params={
                "query": query,
                "mode": "artlist",
                "format": "json",
                "maxrecords": max_records,
                "sort": "DateDesc",
            })
            resp.raise_for_status()
            data = resp.json()

        articles = data.get("articles", [])
        _set_cached(cache_key, articles)
        return articles

    except Exception as e:
        logger.debug(f"GDELT article fetch failed: {e}")
        return None


async def analyze_event(question: str, category: str = "other") -> Optional[GdeltSignal]:
    """Analyze a Polymarket market question using GDELT data.

    Returns a GdeltSignal with volume momentum, tone, and probability adjustment.
    """
    search_terms = _extract_search_terms(question)
    if not search_terms or len(search_terms) < 5:
        return None

    # Add theme filter if available for this category
    themes = CATEGORY_THEMES.get(category, [])
    if themes:
        # Use the first relevant theme
        query = f'{search_terms} theme:{themes[0]}'
    else:
        query = search_terms

    # Fetch volume and tone in parallel
    vol_data, tone_data = await asyncio.gather(
        fetch_volume_timeline(query, "7d"),
        fetch_tone_timeline(query, "7d"),
        return_exceptions=True,
    )

    if isinstance(vol_data, Exception):
        vol_data = None
    if isinstance(tone_data, Exception):
        tone_data = None

    # Calculate volume momentum
    volume_ratio = 1.0
    article_count = 0
    momentum = "stable"

    if vol_data and len(vol_data) >= 2:
        # Get recent vs baseline volume
        volumes = []
        for point in vol_data:
            v = point.get("value", point.get("count", 0))
            if isinstance(v, (int, float)):
                volumes.append(v)

        if volumes and len(volumes) >= 3:
            article_count = int(sum(volumes))
            recent = sum(volumes[-3:]) / 3  # Last 3 periods
            baseline = sum(volumes[:-3]) / max(1, len(volumes) - 3)  # Earlier periods

            if baseline > 0:
                volume_ratio = recent / baseline
            elif recent > 0:
                volume_ratio = 5.0  # New topic with no baseline

            if volume_ratio > 2.0:
                momentum = "rising"
            elif volume_ratio < 0.5:
                momentum = "declining"

    # Calculate tone
    tone = 0.0
    tone_shift = 0.0

    if tone_data and len(tone_data) >= 2:
        tones = []
        for point in tone_data:
            t = point.get("value", point.get("tone", 0))
            if isinstance(t, (int, float)):
                tones.append(t)

        if tones and len(tones) >= 3:
            tone = sum(tones[-3:]) / 3  # Recent average tone
            baseline_tone = sum(tones[:-3]) / max(1, len(tones) - 3)
            tone_shift = tone - baseline_tone

    # Calculate probability adjustment
    prob_adj = 0.0

    # Volume momentum signal
    if volume_ratio > 5.0:
        prob_adj += 0.15  # Major breakout
    elif volume_ratio > 3.0:
        prob_adj += 0.10
    elif volume_ratio > 2.0:
        prob_adj += 0.05
    elif volume_ratio > 1.5:
        prob_adj += 0.02

    # Tone shift signal (direction depends on event type)
    # For conflict/negative events: negative tone = more likely
    # For cooperation/positive events: positive tone = more likely
    if category in ("politics", "other"):
        # Most political/geopolitical markets are about negative events
        if tone_shift < -1.0:
            prob_adj += 0.05  # Sentiment turning more negative = event more likely
        elif tone_shift > 1.0:
            prob_adj -= 0.03
    elif category == "economics":
        # Economic events: tone shift indicates direction
        if tone_shift < -1.5:
            prob_adj += 0.05  # Negative economic sentiment
        elif tone_shift > 1.5:
            prob_adj += 0.03  # Positive economic sentiment (for "will economy grow" type)

    # Cap adjustment
    prob_adj = max(-0.20, min(0.20, prob_adj))

    # Source count from articles
    source_count = 0
    articles = await fetch_article_list(search_terms, max_records=10)
    if articles and not isinstance(articles, Exception):
        domains = set()
        for art in articles:
            url = art.get("url", "")
            if url:
                domain = url.split("/")[2] if len(url.split("/")) > 2 else ""
                domains.add(domain)
        source_count = len(domains)

    reasoning = (
        f"GDELT: vol_ratio={volume_ratio:.1f}x ({momentum}), "
        f"tone={tone:+.1f} (shift={tone_shift:+.1f}), "
        f"{article_count} articles, {source_count} sources"
    )

    if prob_adj != 0:
        logger.info(f"[GDELT] {question[:40]} -> adj={prob_adj:+.2f} | {reasoning}")

    return GdeltSignal(
        volume_ratio=volume_ratio,
        tone=tone,
        tone_shift=tone_shift,
        article_count=article_count,
        source_count=source_count,
        momentum=momentum,
        probability_adjustment=prob_adj,
        reasoning=reasoning,
    )
