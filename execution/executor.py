"""
Executor — Trade Execution Against Polymarket CLOB
"""

import time
import hmac
import hashlib
import json
import logging
from dataclasses import dataclass

import requests

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    success: bool
    order_id: str
    fill_price: float
    filled_size: float
    slippage: float
    error: str = ""


class PaperExecutor:
    """Simulated executor for paper trading."""

    def __init__(self):
        self.order_count = 0
        self.fills: list[dict] = []

    def buy(self, token_id: str, side: str, size_usd: float,
            expected_price: float, market_question: str = "") -> OrderResult:
        self.order_count += 1
        slippage = 0.01
        fill_price = expected_price + slippage
        num_contracts = size_usd / fill_price

        self.fills.append({
            "order_id": f"PAPER-{self.order_count:06d}",
            "token_id": token_id, "side": side,
            "fill_price": fill_price, "size_usd": size_usd,
            "num_contracts": num_contracts, "slippage": slippage,
            "timestamp": time.time(), "market": market_question[:60],
        })

        logger.info(
            f"[PAPER] BUY {side} | {num_contracts:.1f} contracts @ {fill_price:.3f} | "
            f"${size_usd:.2f} | {market_question[:40]}"
        )
        return OrderResult(True, f"PAPER-{self.order_count:06d}",
                          fill_price, num_contracts, slippage)

    def sell(self, token_id: str, side: str, num_contracts: float,
             expected_price: float, market_question: str = "") -> OrderResult:
        self.order_count += 1
        slippage = 0.01
        fill_price = expected_price - slippage

        self.fills.append({
            "order_id": f"PAPER-{self.order_count:06d}",
            "token_id": token_id, "side": f"SELL_{side}",
            "fill_price": fill_price, "num_contracts": num_contracts,
            "slippage": slippage, "timestamp": time.time(),
            "market": market_question[:60],
        })

        logger.info(
            f"[PAPER] SELL {side} | {num_contracts:.1f} contracts @ {fill_price:.3f} | "
            f"{market_question[:40]}"
        )
        return OrderResult(True, f"PAPER-{self.order_count:06d}",
                          fill_price, num_contracts, slippage)


class LiveExecutor:
    """Real executor using Polymarket CLOB API."""

    def __init__(self):
        self.api_key = settings.POLYMARKET_API_KEY
        self.api_secret = settings.POLYMARKET_API_SECRET
        self.passphrase = settings.POLYMARKET_PASSPHRASE
        self.base_url = settings.POLYMARKET_CLOB_URL

    def _sign_request(self, method: str, path: str, body: str = "") -> dict:
        timestamp = str(int(time.time()))
        message = timestamp + method.upper() + path + body
        signature = hmac.new(
            self.api_secret.encode(), message.encode(), hashlib.sha256
        ).hexdigest()
        return {
            "POLY_ADDRESS": self.api_key,
            "POLY_SIGNATURE": signature,
            "POLY_TIMESTAMP": timestamp,
            "POLY_PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
        }

    def buy(self, token_id: str, side: str, size_usd: float,
            expected_price: float, market_question: str = "") -> OrderResult:
        num_contracts = size_usd / expected_price
        order_payload = {
            "tokenID": token_id,
            "price": str(round(expected_price + 0.01, 2)),
            "size": str(round(num_contracts, 2)),
            "side": "BUY", "type": "FOK",
        }
        path = "/order"
        body = json.dumps(order_payload)
        headers = self._sign_request("POST", path, body)

        try:
            resp = requests.post(f"{self.base_url}{path}", headers=headers,
                                data=body, timeout=10)
            resp.raise_for_status()
            result = resp.json()
            fill_price = float(result.get("avgPrice", expected_price))
            filled = float(result.get("filledSize", 0))
            slippage = fill_price - expected_price

            logger.info(f"[LIVE] BUY {side} | {filled:.1f} @ {fill_price:.3f} | slippage={slippage:.3f}")
            return OrderResult(filled > 0, result.get("orderID", ""),
                              fill_price, filled, slippage)
        except requests.RequestException as e:
            logger.error(f"Order failed: {e}")
            return OrderResult(False, "", 0, 0, 0, str(e))

    def sell(self, token_id: str, side: str, num_contracts: float,
             expected_price: float, market_question: str = "") -> OrderResult:
        order_payload = {
            "tokenID": token_id,
            "price": str(round(expected_price - 0.01, 2)),
            "size": str(round(num_contracts, 2)),
            "side": "SELL", "type": "FOK",
        }
        path = "/order"
        body = json.dumps(order_payload)
        headers = self._sign_request("POST", path, body)

        try:
            resp = requests.post(f"{self.base_url}{path}", headers=headers,
                                data=body, timeout=10)
            resp.raise_for_status()
            result = resp.json()
            fill_price = float(result.get("avgPrice", expected_price))
            filled = float(result.get("filledSize", 0))
            slippage = expected_price - fill_price
            return OrderResult(filled > 0, result.get("orderID", ""),
                              fill_price, filled, slippage)
        except requests.RequestException as e:
            logger.error(f"Sell order failed: {e}")
            return OrderResult(False, "", 0, 0, 0, str(e))


def get_executor():
    """Factory — returns paper or live executor based on settings."""
    if settings.LIVE_MODE:
        logger.warning("LIVE TRADING MODE — real money at risk")
        return LiveExecutor()
    else:
        logger.info("Paper trading mode — no real orders")
        return PaperExecutor()
