"""
Geopolitical Estimator — Combines three intelligence sources:
1. GDELT: Global news volume momentum + tone shifts (free, no key)
2. ACLED: Armed conflict/protest data (free with registration)
3. ICEWS: Political interaction analysis via GDELT proxy (free, no key)

This is the most powerful estimator for politics, economics, and geopolitical events.
"""

import asyncio
import logging

from core.market import Market, ProbEstimate
from estimators.base import Estimator
from sources.gdelt import analyze_event, GdeltSignal
from sources.acled import analyze_conflict, AcledSignal
from sources.icews import analyze_interactions, IcewsSignal

logger = logging.getLogger(__name__)

GEOPOLITICAL_CATEGORIES = {"politics", "economics", "other"}


class GeopoliticalEstimator(Estimator):
    """Combines GDELT + ACLED + ICEWS into unified probability estimates."""

    @property
    def source_name(self) -> str:
        return "gdelt_news"

    async def estimate(self, market: Market) -> ProbEstimate | None:
        if market.category not in GEOPOLITICAL_CATEGORIES:
            return None

        # Run all three sources in parallel
        results = await asyncio.gather(
            analyze_event(market.question, market.category),
            analyze_conflict(market.question),
            analyze_interactions(market.question),
            return_exceptions=True,
        )

        gdelt: GdeltSignal | None = results[0] if not isinstance(results[0], Exception) else None
        acled: AcledSignal | None = results[1] if not isinstance(results[1], Exception) else None
        icews: IcewsSignal | None = results[2] if not isinstance(results[2], Exception) else None

        # Need at least one source with data
        if not gdelt and not acled and not icews:
            return None

        # Aggregate probability adjustments with weights
        total_adj = 0.0
        total_weight = 0.0
        sources_used = []
        reasoning_parts = []

        if gdelt and (gdelt.article_count >= 3 or gdelt.volume_ratio > 1.5):
            total_adj += gdelt.probability_adjustment * 0.40
            total_weight += 0.40
            sources_used.append("gdelt")
            reasoning_parts.append(gdelt.reasoning)

        if acled and acled.event_count_7d > 0:
            total_adj += acled.probability_adjustment * 0.35
            total_weight += 0.35
            sources_used.append("acled")
            reasoning_parts.append(acled.reasoning)

        if icews and icews.interaction_count > 0:
            total_adj += icews.probability_adjustment * 0.25
            total_weight += 0.25
            sources_used.append("icews")
            reasoning_parts.append(icews.reasoning)

        if total_weight == 0:
            return None

        # Normalize adjustment
        adj = total_adj / total_weight

        # Apply to market price
        base_prob = market.yes_price
        adjusted = max(0.02, min(0.98, base_prob + adj))

        # Confidence based on source count and data quality
        confidence = 0.25 + 0.15 * len(sources_used)

        if gdelt and gdelt.volume_ratio > 3.0:
            confidence += 0.10
        if gdelt and gdelt.article_count >= 50:
            confidence += 0.10
        if acled and acled.event_count_7d >= 10:
            confidence += 0.10
        if icews and icews.interaction_count >= 20:
            confidence += 0.05

        confidence = min(0.85, confidence)

        detail = "+".join(sources_used)
        reasoning = " | ".join(reasoning_parts[:2])  # Keep it concise

        return ProbEstimate(
            probability=adjusted,
            confidence=confidence,
            source=self.source_name,
            source_detail=detail,
            reasoning=reasoning,
        )
