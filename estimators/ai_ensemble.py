"""
AI Ensemble Estimator — Uses Claude to estimate event probabilities.
The fallback estimator that works for every category.

Key design: present the question FIRST, get AI's estimate, then compare
with market price to avoid anchoring bias.
"""

import re
import time
import logging
from typing import Optional

from config import settings
from core.market import Market, ProbEstimate
from estimators.base import Estimator

logger = logging.getLogger(__name__)

# Track daily spend to stay within budget
_daily_spend_usd = 0.0
_daily_reset_time = 0.0
_COST_PER_CALL = 0.003  # ~$0.003 per Sonnet call


def _check_budget() -> bool:
    """Check if we're within daily AI budget."""
    global _daily_spend_usd, _daily_reset_time
    now = time.time()
    if now - _daily_reset_time > 86400:
        _daily_spend_usd = 0.0
        _daily_reset_time = now
    return _daily_spend_usd < settings.AI_DAILY_BUDGET_USD


def _record_spend():
    global _daily_spend_usd
    _daily_spend_usd += _COST_PER_CALL


class AIEnsembleEstimator(Estimator):
    """Estimates probability using Claude AI reasoning."""

    def __init__(self, api_key: str = None):
        self._client = None
        self._api_key = api_key

    @property
    def source_name(self) -> str:
        return "ai_ensemble"

    def _get_client(self):
        if self._client is None:
            import os
            key = self._api_key or os.environ.get("ANTHROPIC_API_KEY", "").strip() or settings.ANTHROPIC_API_KEY
            print(f"[AI_DEBUG] _api_key={bool(self._api_key)}, env={bool(os.environ.get('ANTHROPIC_API_KEY'))}, settings={bool(settings.ANTHROPIC_API_KEY)}, final_key_len={len(key) if key else 0}", flush=True)
            if not key:
                raise ValueError("ANTHROPIC_API_KEY not configured")
            import anthropic
            self._client = anthropic.Anthropic(api_key=key)
            print(f"[AI_DEBUG] Client created successfully!", flush=True)
        return self._client

    _call_count = 0

    async def estimate(self, market: Market) -> ProbEstimate | None:
        # Only call AI for every 3rd market to speed up scans
        AIEnsembleEstimator._call_count += 1
        if AIEnsembleEstimator._call_count % 3 != 0:
            return None

        try:
            client = self._get_client()
            prompt = self._build_prompt(market)

            # Run sync Claude call in thread pool to avoid blocking event loop
            import asyncio
            loop = asyncio.get_event_loop()
            message = await loop.run_in_executor(
                None,
                lambda: client.messages.create(
                    model=settings.AI_MODEL,
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}],
                ),
            )

            _record_spend()
            response = message.content[0].text
            logger.info(f"[AI] {market.question[:40]} -> {response[:60]}")

            # Parse probability from response
            prob = self._parse_probability(response)
            if prob is None:
                return None

            # Parse confidence
            confidence = self._parse_confidence(response)

            return ProbEstimate(
                probability=prob,
                confidence=confidence,
                source=self.source_name,
                source_detail=settings.AI_MODEL,
                reasoning=response[:200],
            )

        except Exception as e:
            logger.error(f"AI estimation failed: {e}")
            return None

    def _build_prompt(self, market: Market) -> str:
        """Build a probability estimation prompt.

        IMPORTANT: Question is presented first WITHOUT the market price
        to avoid anchoring the model's estimate.
        """
        return f"""You are a calibrated probability forecaster. Estimate the probability that this event resolves YES.

Question: {market.question}

Category: {market.category}
Resolution date: {market.end_date or 'Unknown'}
Today's date: {time.strftime('%Y-%m-%d')}

Instructions:
1. Consider base rates, recent trends, and relevant evidence
2. Account for uncertainty — avoid extreme probabilities unless very confident
3. Be well-calibrated: events you rate at 70% should happen ~70% of the time

Respond in this exact format:
PROBABILITY: 0.XX
CONFIDENCE: high/medium/low
REASONING: [1-2 sentence explanation]"""

    def _parse_probability(self, response: str) -> Optional[float]:
        """Extract probability from AI response."""
        # Look for PROBABILITY: 0.XX pattern
        match = re.search(r'PROBABILITY:\s*(0?\.\d+|1\.0|0|1)', response, re.IGNORECASE)
        if match:
            prob = float(match.group(1))
            return max(0.01, min(0.99, prob))

        # Fallback: look for any decimal between 0 and 1
        match = re.search(r'\b(0\.\d+)\b', response)
        if match:
            prob = float(match.group(1))
            if 0.01 <= prob <= 0.99:
                return prob

        # Fallback: look for percentage
        match = re.search(r'(\d{1,2})%', response)
        if match:
            return int(match.group(1)) / 100

        return None

    def _parse_confidence(self, response: str) -> float:
        """Extract confidence level from AI response."""
        response_lower = response.lower()
        if "high" in response_lower and "confidence" in response_lower:
            return 0.80
        elif "low" in response_lower and "confidence" in response_lower:
            return 0.40
        elif "medium" in response_lower and "confidence" in response_lower:
            return 0.60
        return 0.55  # Default moderate confidence
