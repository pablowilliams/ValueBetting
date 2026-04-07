"""
Cross-Market Estimator — Compares prices across prediction platforms.
The single most reliable signal: if Polymarket says 40% but Metaculus says 55%,
that 15-point gap is highly informative.
"""

import asyncio
import logging

from core.market import Market, ProbEstimate
from core.matcher import match_score, is_good_match
from estimators.base import Estimator
from sources import manifold, metaculus

logger = logging.getLogger(__name__)

# Minimum match score to consider a cross-platform question as the same event
MIN_MATCH_THRESHOLD = 0.35


class CrossMarketEstimator(Estimator):
    """Estimates probability by checking the same event on other platforms."""

    @property
    def source_name(self) -> str:
        return "cross_market"

    async def estimate(self, market: Market) -> ProbEstimate | None:
        """Search Manifold and Metaculus for matching markets."""
        question = market.question

        # Search Manifold only (Metaculus rate-limits aggressively)
        results = await asyncio.gather(
            self._check_manifold(question),
            return_exceptions=True,
        )

        probs = []
        sources = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                continue
            if result is not None:
                prob, source = result
                probs.append(prob)
                sources.append(source)

        if not probs:
            return None

        # Use median if multiple platforms, single value otherwise
        probs.sort()
        if len(probs) == 1:
            median_prob = probs[0]
            confidence = 0.65  # Single cross-market source
        else:
            median_prob = probs[len(probs) // 2]
            # Agreement boosts confidence
            spread = max(probs) - min(probs)
            confidence = 0.85 - spread  # High confidence if platforms agree

        confidence = max(0.3, min(0.95, confidence))

        return ProbEstimate(
            probability=median_prob,
            confidence=confidence,
            source=self.source_name,
            source_detail=", ".join(sources),
            reasoning=f"Cross-market: {', '.join(f'{s}={p:.2f}' for s, p in zip(sources, probs))}",
        )

    async def _check_manifold(self, question: str) -> tuple[float, str] | None:
        """Search Manifold Markets for a matching question."""
        try:
            result = await manifold.find_matching_probability(question)
            if result:
                prob, url = result
                return (prob, "manifold")
        except Exception as e:
            logger.debug(f"Manifold check failed: {e}")
        return None

    async def _check_metaculus(self, question: str) -> tuple[float, str] | None:
        """Search Metaculus for a matching question."""
        try:
            result = await metaculus.find_matching_probability(question)
            if result:
                prob, url = result
                return (prob, "metaculus")
        except Exception as e:
            logger.debug(f"Metaculus check failed: {e}")
        return None
