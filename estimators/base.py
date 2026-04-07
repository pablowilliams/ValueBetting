"""Abstract base for probability estimators."""

from abc import ABC, abstractmethod

from core.market import Market, ProbEstimate


class Estimator(ABC):
    """Base class for probability estimators.

    Each estimator takes a Market and returns a ProbEstimate
    representing the estimated true probability of YES.
    """

    @abstractmethod
    async def estimate(self, market: Market) -> ProbEstimate | None:
        """Estimate the true probability for this market.

        Returns None if the estimator has no data for this market.
        """
        ...

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Identifier for this estimator (matches category source_weights keys)."""
        ...
