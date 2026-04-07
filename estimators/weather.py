"""
Weather Estimator — Uses Open-Meteo ensemble forecasts.
No API key required. Free and unlimited.
"""

import logging

from core.market import Market, ProbEstimate
from estimators.base import Estimator
from sources.weather import estimate_temp_probability

logger = logging.getLogger(__name__)


class WeatherEstimator(Estimator):
    """Estimates probability for weather markets using forecast models."""

    @property
    def source_name(self) -> str:
        return "weather_model"

    async def estimate(self, market: Market) -> ProbEstimate | None:
        if market.category != "weather":
            return None

        result = await estimate_temp_probability(market.question)
        if result is None:
            return None

        prob, confidence, reasoning = result

        return ProbEstimate(
            probability=prob,
            confidence=confidence,
            source=self.source_name,
            source_detail="open-meteo",
            reasoning=reasoning,
        )
