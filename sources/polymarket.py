"""
Polymarket Feed — Market Scanner
Fetches all active markets from Polymarket's Gamma API.
Retrieves orderbook data from CLOB API.
"""

import time
import logging
from typing import Optional

import httpx

from config import settings
from core.market import Market

logger = logging.getLogger(__name__)


class PolymarketScanner:
    """Fetches and caches Polymarket markets."""

    def __init__(self):
        self.gamma_url = settings.POLYMARKET_GAMMA_URL
        self.clob_url = settings.POLYMARKET_CLOB_URL
        self.cache: dict[str, Market] = {}
        self._last_fetch = 0.0

    async def fetch_active_markets(self, limit: int = 100) -> list[Market]:
        """Fetch active markets from Gamma API."""
        markets = []
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{self.gamma_url}/markets",
                    params={
                        "active": "true",
                        "closed": "false",
                        "limit": limit,
                        "order": "volume",
                        "ascending": "false",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            now = time.time()
            market_list = data if isinstance(data, list) else data.get("data", data.get("markets", []))

            for m in market_list:
                market = self._parse_market(m, now)
                if market:
                    markets.append(market)
                    self.cache[market.condition_id] = market

            self._last_fetch = now
            logger.info(f"Fetched {len(markets)} active markets from Polymarket")

        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch Polymarket markets: {e}")

        return markets

    def _parse_market(self, data: dict, timestamp: float) -> Optional[Market]:
        """Parse a market from Gamma/CLOB API response."""
        try:
            condition_id = data.get("condition_id", data.get("id", ""))
            question = data.get("question", data.get("title", ""))

            if not condition_id or not question:
                return None

            tokens = data.get("tokens", [])
            token_id_yes = ""
            token_id_no = ""
            yes_price = 0.0
            no_price = 0.0

            if tokens and len(tokens) >= 2:
                for token in tokens:
                    outcome = token.get("outcome", "").upper()
                    if outcome == "YES":
                        token_id_yes = token.get("token_id", "")
                        yes_price = float(token.get("price", 0))
                    elif outcome == "NO":
                        token_id_no = token.get("token_id", "")
                        no_price = float(token.get("price", 0))
            else:
                prices = data.get("outcomePrices", "")
                if prices:
                    import json as _json
                    price_list = _json.loads(prices) if isinstance(prices, str) else prices
                    if len(price_list) >= 2:
                        yes_price = float(price_list[0])
                        no_price = float(price_list[1])
                else:
                    yes_price = float(data.get("yes_price", 0.5))
                    no_price = 1.0 - yes_price

            # Extract token IDs from clobTokenIds if not in tokens
            if not token_id_yes:
                clob_ids = data.get("clobTokenIds", "")
                if clob_ids:
                    import json as _json
                    id_list = _json.loads(clob_ids) if isinstance(clob_ids, str) else clob_ids
                    if len(id_list) >= 2:
                        token_id_yes = id_list[0]
                        token_id_no = id_list[1]

            return Market(
                condition_id=condition_id,
                question=question,
                token_id_yes=token_id_yes,
                token_id_no=token_id_no,
                yes_price=yes_price,
                no_price=no_price,
                volume=float(data.get("volume", data.get("volumeNum", 0))),
                liquidity=float(data.get("liquidity", data.get("liquidityNum", 0))),
                end_date=data.get("end_date_iso", data.get("endDate", "")),
                category=data.get("category", data.get("groupSlug", "")),
                spread=0.0,
                best_bid_yes=0.0,
                best_ask_yes=0.0,
                orderbook_depth=0.0,
                timestamp=timestamp,
                raw=data,
            )
        except (ValueError, KeyError) as e:
            logger.debug(f"Failed to parse market: {e}")
            return None

    async def enrich_with_orderbook(self, market: Market) -> Market:
        """Add orderbook data (spread, depth) to a market."""
        if not market.token_id_yes:
            return market

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self.clob_url}/book",
                    params={"token_id": market.token_id_yes},
                )
                resp.raise_for_status()
                book = resp.json()

            bids = book.get("bids", [])
            asks = book.get("asks", [])

            market.best_bid_yes = float(bids[0]["price"]) if bids else 0.0
            market.best_ask_yes = float(asks[0]["price"]) if asks else 0.0
            market.spread = market.best_ask_yes - market.best_bid_yes

            depth = 0.0
            for ask in asks:
                price = float(ask["price"])
                if price <= market.best_ask_yes + 0.02:
                    depth += float(ask["size"]) * price
            market.orderbook_depth = depth

        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch orderbook for {market.condition_id}: {e}")

        return market

    def get_market(self, condition_id: str) -> Optional[Market]:
        """Get a cached market by condition ID."""
        return self.cache.get(condition_id)
