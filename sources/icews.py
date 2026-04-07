"""
ICEWS — Integrated Crisis Early Warning System.
Curated political event data from Harvard Dataverse.

ICEWS is higher quality than GDELT (human-curated) but updated less frequently.
Uses CAMEO event codes to track political interactions between actors.

Key value: tracks escalation/de-escalation patterns between countries.
Best for: "Will country X take action against Y?" type markets.

Data access: Harvard Dataverse (free download, updated monthly).
For real-time use, we use the GDELT event database as a proxy with
ICEWS-style analysis (GoldsteinScale tracking for actor pairs).
"""

import asyncio
import logging
import time
from typing import Optional
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

# We use GDELT's event data as a real-time ICEWS proxy
# (ICEWS itself updates monthly, too slow for trading)
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# Rate limiting (shared with GDELT)
_last_request = 0.0
_MIN_INTERVAL = 6.0
_cache: dict[str, tuple[float, dict]] = {}
_CACHE_TTL = 900.0


# Actor pairs commonly traded on Polymarket
ACTOR_PAIRS = {
    ("russia", "ukraine"): "Russia-Ukraine conflict",
    ("israel", "iran"): "Israel-Iran tensions",
    ("israel", "palestine"): "Israel-Palestine conflict",
    ("china", "taiwan"): "China-Taiwan tensions",
    ("us", "china"): "US-China relations",
    ("us", "iran"): "US-Iran relations",
    ("us", "russia"): "US-Russia relations",
    ("us", "north korea"): "US-North Korea",
    ("india", "pakistan"): "India-Pakistan tensions",
}

# CAMEO-style escalation keywords (mapped from ICEWS coding)
COOPERATION_KEYWORDS = [
    "negotiate", "agreement", "deal", "treaty", "ceasefire",
    "diplomatic", "talks", "peace", "cooperat", "ally",
]
CONFLICT_KEYWORDS = [
    "sanction", "threaten", "attack", "military", "troops",
    "strike", "missile", "bomb", "invasion", "war",
    "escalat", "retaliat", "clash", "confront",
]


@dataclass
class IcewsSignal:
    """Political interaction signal from ICEWS-style analysis."""
    actor_pair: str
    cooperation_score: float    # 0-1: how cooperative recent interactions are
    conflict_score: float       # 0-1: how conflictual recent interactions are
    escalation_trend: str       # "escalating", "stable", "de-escalating"
    interaction_count: int
    probability_adjustment: float
    reasoning: str


def _extract_actor_pair(question: str) -> Optional[tuple[str, str]]:
    """Extract an actor pair from a market question."""
    q = question.lower()
    for (a1, a2), desc in ACTOR_PAIRS.items():
        if a1 in q and a2 in q:
            return (a1, a2)
        # Check single actor for unilateral action markets
        if a1 in q or a2 in q:
            # "Will Russia..." or "Will Iran..."
            for (pa1, pa2), _ in ACTOR_PAIRS.items():
                if pa1 in q or pa2 in q:
                    return (pa1, pa2)
    return None


async def _fetch_interaction_articles(actor1: str, actor2: str) -> Optional[list[dict]]:
    """Fetch recent news about interactions between two actors."""
    cache_key = f"icews:{actor1}:{actor2}"
    if cache_key in _cache:
        ts, data = _cache[cache_key]
        if time.time() - ts < _CACHE_TTL:
            return data

    global _last_request
    now = time.time()
    if now - _last_request < _MIN_INTERVAL:
        await asyncio.sleep(_MIN_INTERVAL - (now - _last_request))
    _last_request = time.time()

    query = f'"{actor1}" "{actor2}"'

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(GDELT_DOC_API, params={
                "query": query,
                "mode": "artlist",
                "format": "json",
                "maxrecords": 50,
                "sort": "DateDesc",
                "timespan": "7d",
            })
            resp.raise_for_status()
            data = resp.json()

        articles = data.get("articles", [])
        _cache[cache_key] = (time.time(), articles)
        return articles

    except Exception as e:
        logger.debug(f"ICEWS proxy fetch failed: {e}")
        return None


async def analyze_interactions(question: str) -> Optional[IcewsSignal]:
    """Analyze political interactions for a market question."""
    pair = _extract_actor_pair(question)
    if not pair:
        return None

    actor1, actor2 = pair
    articles = await _fetch_interaction_articles(actor1, actor2)

    if not articles:
        return None

    # Analyze cooperation vs conflict in article titles/tones
    coop_count = 0
    conflict_count = 0
    total_tone = 0.0

    for art in articles:
        title = (art.get("title", "") or "").lower()
        tone = art.get("tone", 0)
        if isinstance(tone, (int, float)):
            total_tone += tone

        for kw in COOPERATION_KEYWORDS:
            if kw in title:
                coop_count += 1
                break

        for kw in CONFLICT_KEYWORDS:
            if kw in title:
                conflict_count += 1
                break

    total = max(1, coop_count + conflict_count)
    cooperation_score = coop_count / total
    conflict_score = conflict_count / total
    avg_tone = total_tone / max(1, len(articles))

    # Determine escalation trend
    if conflict_score > 0.6 and avg_tone < -3:
        trend = "escalating"
    elif cooperation_score > 0.6 and avg_tone > -1:
        trend = "de-escalating"
    else:
        trend = "stable"

    # Probability adjustment
    prob_adj = 0.0
    q_lower = question.lower()

    # For conflict/attack/invasion questions
    is_conflict_question = any(w in q_lower for w in [
        "invade", "attack", "strike", "war", "military", "nuclear",
        "sanction", "embargo",
    ])
    # For cooperation/deal/peace questions
    is_peace_question = any(w in q_lower for w in [
        "deal", "peace", "ceasefire", "negotiate", "treaty", "agree",
    ])

    if is_conflict_question:
        if trend == "escalating":
            prob_adj += 0.08
        elif trend == "de-escalating":
            prob_adj -= 0.05
    elif is_peace_question:
        if trend == "de-escalating":
            prob_adj += 0.08
        elif trend == "escalating":
            prob_adj -= 0.05

    # Strong tone shift amplifies signal
    if abs(avg_tone) > 5:
        prob_adj *= 1.3

    prob_adj = max(-0.15, min(0.15, prob_adj))

    pair_name = ACTOR_PAIRS.get((actor1, actor2), f"{actor1}-{actor2}")
    reasoning = (
        f"ICEWS-proxy {pair_name}: coop={cooperation_score:.0%} conflict={conflict_score:.0%}, "
        f"tone={avg_tone:+.1f}, {len(articles)} articles, trend={trend}"
    )

    if prob_adj != 0:
        logger.info(f"[ICEWS] {question[:40]} -> adj={prob_adj:+.2f} | {reasoning}")

    return IcewsSignal(
        actor_pair=pair_name,
        cooperation_score=cooperation_score,
        conflict_score=conflict_score,
        escalation_trend=trend,
        interaction_count=len(articles),
        probability_adjustment=prob_adj,
        reasoning=reasoning,
    )
