"""
ACLED — Armed Conflict Location & Event Data.
Real-time conflict, protest, and political violence monitoring.

Free API access with registration at https://acleddata.com/
Provides: battles, protests, riots, violence against civilians,
strategic developments, explosions/remote violence.

Key value: ACLED is curated and verified (unlike GDELT which is raw news).
Best for: geopolitical events, military actions, protest predictions.
"""

import asyncio
import logging
import time
from typing import Optional
from dataclasses import dataclass

import httpx

from config import settings

logger = logging.getLogger(__name__)

# ACLED API (requires free registration for key + email)
ACLED_API = "https://api.acleddata.com/acled/read"

# Cache
_cache: dict[str, tuple[float, list]] = {}
_CACHE_TTL = 1800.0  # 30 minutes (ACLED updates daily)


# ACLED event types
EVENT_TYPES = {
    "battles": "Battles",
    "protests": "Protests",
    "riots": "Riots",
    "violence_civilians": "Violence against civilians",
    "strategic": "Strategic developments",
    "explosions": "Explosions/Remote violence",
}


@dataclass
class AcledSignal:
    """Conflict/protest signal from ACLED data."""
    event_count_7d: int         # Events in last 7 days
    event_count_30d: int        # Events in last 30 days
    fatalities_7d: int          # Fatalities in last 7 days
    trend: str                  # "escalating", "stable", "de-escalating"
    dominant_event_type: str    # Most common event type
    probability_adjustment: float
    reasoning: str


async def fetch_events(
    country: str = "",
    event_type: str = "",
    days_back: int = 30,
    limit: int = 500,
) -> Optional[list[dict]]:
    """Fetch recent events from ACLED API."""
    # Check for API key
    acled_key = getattr(settings, "ACLED_API_KEY", "") or ""
    acled_email = getattr(settings, "ACLED_EMAIL", "") or ""

    if not acled_key or not acled_email:
        # ACLED requires registration — return None if no credentials
        return None

    cache_key = f"acled:{country}:{event_type}:{days_back}"
    if cache_key in _cache:
        ts, data = _cache[cache_key]
        if time.time() - ts < _CACHE_TTL:
            return data

    from datetime import datetime, timedelta
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    params = {
        "key": acled_key,
        "email": acled_email,
        "event_date": f"{start_date}|{end_date}",
        "event_date_where": "BETWEEN",
        "limit": limit,
    }

    if country:
        params["country"] = country
    if event_type:
        params["event_type"] = event_type

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(ACLED_API, params=params)
            resp.raise_for_status()
            data = resp.json()

        events = data.get("data", [])
        _cache[cache_key] = (time.time(), events)
        logger.info(f"[ACLED] Fetched {len(events)} events for {country or 'global'}")
        return events

    except Exception as e:
        logger.debug(f"ACLED fetch failed: {e}")
        return None


def _extract_country(question: str) -> Optional[str]:
    """Extract country name from a market question."""
    # Common countries in Polymarket geopolitical events
    countries = {
        "russia": "Russia", "ukraine": "Ukraine", "china": "China",
        "iran": "Iran", "israel": "Israel", "gaza": "Palestine",
        "palestine": "Palestine", "taiwan": "Taiwan", "mexico": "Mexico",
        "north korea": "North Korea", "syria": "Syria", "yemen": "Yemen",
        "myanmar": "Myanmar", "sudan": "Sudan", "ethiopia": "Ethiopia",
        "turkey": "Turkey", "india": "India", "pakistan": "Pakistan",
        "afghanistan": "Afghanistan", "iraq": "Iraq", "lebanon": "Lebanon",
        "venezuela": "Venezuela", "brazil": "Brazil", "nigeria": "Nigeria",
        "south africa": "South Africa", "united states": "United States",
    }

    q = question.lower()
    for key, name in countries.items():
        if key in q:
            return name
    return None


async def analyze_conflict(question: str) -> Optional[AcledSignal]:
    """Analyze a market question using ACLED conflict data."""
    country = _extract_country(question)
    if not country:
        return None

    # Fetch 30 days of events
    events = await fetch_events(country=country, days_back=30)
    if events is None:
        return None

    if not events:
        return AcledSignal(
            event_count_7d=0, event_count_30d=0, fatalities_7d=0,
            trend="stable", dominant_event_type="none",
            probability_adjustment=0.0,
            reasoning=f"ACLED: No events in {country} last 30 days",
        )

    # Split into 7-day and 30-day windows
    from datetime import datetime, timedelta
    now = datetime.now()
    week_ago = now - timedelta(days=7)

    events_7d = []
    events_30d = events
    fatalities_7d = 0

    for e in events:
        try:
            event_date = datetime.strptime(e.get("event_date", ""), "%Y-%m-%d")
            if event_date >= week_ago:
                events_7d.append(e)
                fatalities_7d += int(e.get("fatalities", 0))
        except (ValueError, TypeError):
            pass

    count_7d = len(events_7d)
    count_30d = len(events_30d)

    # Trend: compare last 7 days to weekly average over 30 days
    weekly_avg = count_30d / 4.3  # ~4.3 weeks in 30 days
    if weekly_avg > 0:
        ratio = count_7d / weekly_avg
        if ratio > 1.5:
            trend = "escalating"
        elif ratio < 0.5:
            trend = "de-escalating"
        else:
            trend = "stable"
    else:
        trend = "stable" if count_7d == 0 else "escalating"

    # Dominant event type
    type_counts: dict[str, int] = {}
    for e in events_7d:
        etype = e.get("event_type", "unknown")
        type_counts[etype] = type_counts.get(etype, 0) + 1
    dominant = max(type_counts, key=type_counts.get) if type_counts else "none"

    # Probability adjustment
    prob_adj = 0.0
    if trend == "escalating":
        prob_adj += 0.08
        if fatalities_7d > 50:
            prob_adj += 0.07
        elif fatalities_7d > 10:
            prob_adj += 0.04
    elif trend == "de-escalating":
        prob_adj -= 0.05

    # For "will X invade/attack/strike" type questions
    q_lower = question.lower()
    if any(w in q_lower for w in ["invade", "attack", "strike", "military action", "war"]):
        if "Battles" in type_counts or "Explosions" in str(type_counts):
            prob_adj += 0.05

    prob_adj = max(-0.15, min(0.20, prob_adj))

    reasoning = (
        f"ACLED {country}: {count_7d} events (7d), {count_30d} events (30d), "
        f"{fatalities_7d} fatalities (7d), trend={trend}, dominant={dominant}"
    )

    if prob_adj != 0:
        logger.info(f"[ACLED] {question[:40]} -> adj={prob_adj:+.2f} | {reasoning}")

    return AcledSignal(
        event_count_7d=count_7d,
        event_count_30d=count_30d,
        fatalities_7d=fatalities_7d,
        trend=trend,
        dominant_event_type=dominant,
        probability_adjustment=prob_adj,
        reasoning=reasoning,
    )
