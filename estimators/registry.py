"""
Estimator Registry — Maps categories to estimator pipelines.
"""

import logging

from config.categories import get_category_config
from core.market import Market, ProbEstimate
from estimators.base import Estimator
from estimators.cross_market import CrossMarketEstimator
from estimators.ai_ensemble import AIEnsembleEstimator
from estimators.sports import SportsEstimator
from estimators.weather import WeatherEstimator
from estimators.crypto import CryptoEstimator
from estimators.geopolitical import GeopoliticalEstimator

logger = logging.getLogger(__name__)

# Singleton estimator instances
_estimators: dict[str, Estimator] = {}


def _get_estimator(name: str) -> Estimator | None:
    """Get or create an estimator by name."""
    if name not in _estimators:
        if name == "cross_market":
            _estimators[name] = CrossMarketEstimator()
        elif name == "ai_ensemble":
            import os
            key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not key:
                from config import settings as _s
                key = _s.ANTHROPIC_API_KEY
            _estimators[name] = AIEnsembleEstimator(api_key=key)
        elif name == "sports":
            _estimators[name] = SportsEstimator()
        elif name == "weather":
            _estimators[name] = WeatherEstimator()
        elif name == "crypto":
            _estimators[name] = CryptoEstimator()
        elif name == "gdelt_news":
            _estimators[name] = GeopoliticalEstimator()
        else:
            logger.debug(f"No estimator registered for '{name}'")
            return None
    return _estimators[name]


async def run_estimators(market: Market) -> list[ProbEstimate]:
    """Run the appropriate estimator pipeline for a market's category.

    Returns list of ProbEstimates from all estimators that produced data.
    """
    cat_config = get_category_config(market.category)
    estimator_names = cat_config["estimators"]

    estimates = []
    for name in estimator_names:
        estimator = _get_estimator(name)
        if estimator is None:
            continue

        try:
            result = await estimator.estimate(market)
            if result is not None:
                estimates.append(result)
                logger.debug(
                    f"  {name}: prob={result.probability:.3f} "
                    f"conf={result.confidence:.2f}"
                )
        except Exception as e:
            logger.warning(f"Estimator '{name}' failed for {market.question[:40]}: {e}")

    return estimates
