"""
Metaculus API — Superforecaster probability estimates.
Public API for reading questions and community predictions.
Rate-limited to avoid 429 errors (max 5 req/min).
"""

import asyncio
import logging
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://www.metaculus.com/api"

# Rate limiting — Metaculus free tier is strict
_last_request_time = 0.0
_MIN_INTERVAL = 12.0  # seconds between requests (5 req/min)
_consecutive_errors = 0
_backoff_until = 0.0


async def _rate_limit():
    """Enforce rate limiting for Metaculus API."""
    global _last_request_time, _consecutive_errors, _backoff_until

    now = time.time()

    # If in backoff period, skip entirely
    if now < _backoff_until:
        return False

    # Enforce minimum interval
    elapsed = now - _last_request_time
    if elapsed < _MIN_INTERVAL:
        await asyncio.sleep(_MIN_INTERVAL - elapsed)

    _last_request_time = time.time()
    return True


def _handle_error():
    """Track consecutive errors and apply exponential backoff."""
    global _consecutive_errors, _backoff_until
    _consecutive_errors += 1
    if _consecutive_errors >= 3:
        backoff = min(300, 30 * (2 ** (_consecutive_errors - 3)))
        _backoff_until = time.time() + backoff
        logger.warning(f"Metaculus: {_consecutive_errors} errors, backing off {backoff}s")


def _handle_success():
    global _consecutive_errors
    _consecutive_errors = 0


async def search_questions(query: str, limit: int = 5) -> list[dict]:
    """Search Metaculus questions by query string."""
    if not await _rate_limit():
        return []

    try:
        headers = {"User-Agent": "ValueBetting/0.1 (prediction market research)"}
        async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
            resp = await client.get(
                f"{BASE_URL}/questions/",
                params={
                    "search": query,
                    "limit": limit,
                    "status": "open",
                    "type": "forecast",
                    "order_by": "-activity",
                },
            )
            resp.raise_for_status()
            _handle_success()
            data = resp.json()
            return data.get("results", [])
    except httpx.HTTPError as e:
        _handle_error()
        logger.debug(f"Metaculus search failed: {e}")
        return []


async def get_question_prediction(question_id: int) -> Optional[float]:
    """Get the community prediction for a Metaculus question."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{BASE_URL}/questions/{question_id}/")
            resp.raise_for_status()
            data = resp.json()

            # Community prediction (aggregated forecast)
            prediction = data.get("community_prediction", {})
            if isinstance(prediction, dict):
                return prediction.get("full", {}).get("q2")  # Median
            elif isinstance(prediction, (int, float)):
                return float(prediction)
            return None
    except httpx.HTTPError as e:
        logger.error(f"Metaculus question fetch failed: {e}")
        return None


async def find_matching_probability(question: str) -> Optional[tuple[float, str]]:
    """Search Metaculus for a matching question and return its prediction.

    Returns (probability, url) or None if no match found.
    """
    results = await search_questions(question, limit=3)

    if not results:
        return None

    best = results[0]
    q_id = best.get("id")
    title = best.get("title", "")
    url = best.get("url", f"https://www.metaculus.com/questions/{q_id}/")

    prediction = best.get("community_prediction")
    if prediction is None and q_id:
        prediction = await get_question_prediction(q_id)

    if isinstance(prediction, dict):
        prediction = prediction.get("full", {}).get("q2")

    if prediction is not None:
        prob = float(prediction)
        logger.info(f"Metaculus match: '{title[:60]}' -> {prob:.3f}")
        return (prob, url)

    return None
