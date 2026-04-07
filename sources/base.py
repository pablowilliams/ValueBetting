"""Abstract base class for data sources."""

from abc import ABC, abstractmethod
from typing import Any


class DataSource(ABC):
    """Base class for all external data sources."""

    @abstractmethod
    async def fetch(self, **kwargs) -> Any:
        """Fetch data from the source."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Source identifier."""
        ...
