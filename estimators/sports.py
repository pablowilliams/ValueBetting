"""
Sports Estimator — Uses sportsbook consensus odds as true probability.
Sportsbooks are the gold standard for sports events.
"""

import logging
from typing import Optional

from core.market import Market, ProbEstimate
from estimators.base import Estimator
from sources.odds_api import OddsFeed, OddsEvent

logger = logging.getLogger(__name__)

# Team name extraction patterns
TEAM_ALIASES = {
    "lakers": "los angeles lakers", "celtics": "boston celtics",
    "warriors": "golden state warriors", "nets": "brooklyn nets",
    "knicks": "new york knicks", "sixers": "philadelphia 76ers",
    "bucks": "milwaukee bucks", "heat": "miami heat",
    "suns": "phoenix suns", "mavs": "dallas mavericks",
    "nuggets": "denver nuggets", "clippers": "la clippers",
    "bulls": "chicago bulls", "thunder": "oklahoma city thunder",
    "chiefs": "kansas city chiefs", "eagles": "philadelphia eagles",
    "bills": "buffalo bills", "cowboys": "dallas cowboys",
    "yankees": "new york yankees", "dodgers": "los angeles dodgers",
    "astros": "houston astros", "braves": "atlanta braves",
}


class SportsEstimator(Estimator):
    """Estimates probability using sportsbook consensus."""

    def __init__(self):
        self.odds_feed = OddsFeed()
        self._events: list[OddsEvent] = []
        self._fetched = False

    @property
    def source_name(self) -> str:
        return "sports_odds"

    async def estimate(self, market: Market) -> ProbEstimate | None:
        if market.category != "sports":
            return None

        # Fetch odds if not already done
        if not self._fetched:
            self._events = await self.odds_feed.fetch_all_sports()
            self._fetched = True

        # Find matching event
        match = self._find_matching_event(market.question)
        if match is None:
            return None

        event, side = match

        if side == "home":
            prob = event.home_prob
        else:
            prob = event.away_prob

        return ProbEstimate(
            probability=prob,
            confidence=min(0.90, 0.60 + event.num_books * 0.06),  # More books = more confidence
            source=self.source_name,
            source_detail=f"{event.home_team} vs {event.away_team} ({event.num_books} books)",
            reasoning=f"Sportsbook consensus: {side} {prob:.3f} from {event.num_books} books",
        )

    def _find_matching_event(self, question: str) -> Optional[tuple[OddsEvent, str]]:
        """Find the best matching odds event for a market question."""
        question_lower = question.lower()
        best_event = None
        best_side = "home"
        best_score = 0.0

        for event in self._events:
            if not event.is_fresh:
                continue

            home_score = self._team_in_text(event.home_team, question_lower)
            away_score = self._team_in_text(event.away_team, question_lower)

            if home_score > away_score and home_score > best_score:
                best_event = event
                best_side = "home"
                best_score = home_score
            elif away_score > home_score and away_score > best_score:
                best_event = event
                best_side = "away"
                best_score = away_score

        if best_event and best_score >= 0.7:
            return (best_event, best_side)
        return None

    def _team_in_text(self, team: str, text: str) -> float:
        """Check how well a team name appears in text. Returns 0-1 score."""
        team_lower = team.lower()
        if team_lower in text:
            return 1.0

        # Check aliases
        for alias, full in TEAM_ALIASES.items():
            if full == team_lower and alias in text:
                return 0.90

        # Check last word (mascot/nickname)
        words = team_lower.split()
        if len(words) >= 2 and words[-1] in text:
            return 0.85

        return 0.0

    def reset(self):
        """Reset fetch state for next scan cycle."""
        self._fetched = False
        self._events = []
