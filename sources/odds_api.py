"""
Odds Feed — The Odds API Integration
Pulls sportsbook odds from DraftKings, FanDuel, BetMGM etc.
Converts to implied probability, averages across books.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import httpx

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class OddsEvent:
    """A sports event with sportsbook-derived true probability."""
    event_id: str
    sport: str
    home_team: str
    away_team: str
    commence_time: str
    home_prob: float
    away_prob: float
    draw_prob: float
    num_books: int
    timestamp: float
    raw_odds: dict = field(default_factory=dict)

    @property
    def is_fresh(self) -> bool:
        return (time.time() - self.timestamp) < settings.STALE_ODDS_SECONDS


def american_to_prob(american: int) -> float:
    if american > 0:
        return 100.0 / (american + 100.0)
    else:
        return abs(american) / (abs(american) + 100.0)


def remove_vig(probs: list[float]) -> list[float]:
    total = sum(probs)
    if total == 0:
        return probs
    return [p / total for p in probs]


class OddsFeed:
    """Fetches sportsbook odds from The Odds API."""

    def __init__(self):
        self.cache: dict[str, OddsEvent] = {}
        self._last_fetch: dict[str, float] = {}
        self._requests_remaining: Optional[int] = None

    @property
    def api_quota_remaining(self) -> Optional[int]:
        return self._requests_remaining

    async def fetch_odds(self, sport: str) -> list[OddsEvent]:
        last = self._last_fetch.get(sport, 0)
        if time.time() - last < settings.ODDS_REFRESH_SECONDS:
            return [e for e in self.cache.values() if e.sport == sport]

        if not settings.ODDS_API_KEY:
            logger.error("ODDS_API_KEY not set")
            return []

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{settings.ODDS_API_BASE}/sports/{sport}/odds",
                    params={
                        "apiKey": settings.ODDS_API_KEY,
                        "regions": "us,us2",
                        "markets": "h2h",
                        "oddsFormat": "american",
                        "bookmakers": ",".join(settings.BOOKMAKERS),
                    },
                )
                self._requests_remaining = resp.headers.get("x-requests-remaining")
                if self._requests_remaining:
                    self._requests_remaining = int(self._requests_remaining)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch odds for {sport}: {e}")
            return []

        events = []
        now = time.time()
        for event_data in data:
            event = self._parse_event(event_data, sport, now)
            if event:
                events.append(event)
                self.cache[event.event_id] = event

        self._last_fetch[sport] = now
        logger.info(f"Fetched {len(events)} events for {sport}")
        return events

    def _parse_event(self, data: dict, sport: str, timestamp: float) -> Optional[OddsEvent]:
        event_id = data.get("id", "")
        home_team = data.get("home_team", "")
        away_team = data.get("away_team", "")
        bookmakers = data.get("bookmakers", [])
        if not bookmakers:
            return None

        home_probs, away_probs, draw_probs = [], [], []
        raw_odds = {}

        for book in bookmakers:
            book_key = book.get("key", "")
            for market in book.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                if home_team in outcomes and away_team in outcomes:
                    h = american_to_prob(outcomes[home_team])
                    a = american_to_prob(outcomes[away_team])
                    d = american_to_prob(outcomes.get("Draw", 0)) if "Draw" in outcomes else 0.0
                    if d > 0:
                        h, a, d = remove_vig([h, a, d])
                    else:
                        h, a = remove_vig([h, a])
                    home_probs.append(h)
                    away_probs.append(a)
                    draw_probs.append(d)
                    raw_odds[book_key] = outcomes

        if not home_probs:
            return None

        return OddsEvent(
            event_id=event_id, sport=sport,
            home_team=home_team, away_team=away_team,
            commence_time=data.get("commence_time", ""),
            home_prob=sum(home_probs) / len(home_probs),
            away_prob=sum(away_probs) / len(away_probs),
            draw_prob=sum(draw_probs) / len(draw_probs) if draw_probs else 0.0,
            num_books=len(home_probs),
            timestamp=timestamp, raw_odds=raw_odds,
        )

    async def fetch_all_sports(self) -> list[OddsEvent]:
        all_events = []
        for sport in settings.SPORTS:
            events = await self.fetch_odds(sport)
            all_events.extend(events)
        return all_events

    def get_event(self, event_id: str) -> Optional[OddsEvent]:
        event = self.cache.get(event_id)
        if event and event.is_fresh:
            return event
        return None
