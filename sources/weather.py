"""
Open-Meteo Weather API — Free, no API key required.
Uses ensemble forecast models to estimate probability of temperature thresholds.
"""

import re
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.open-meteo.com/v1/forecast"

# City name -> (latitude, longitude)
CITY_COORDS = {
    "new york": (40.71, -74.01), "nyc": (40.71, -74.01),
    "los angeles": (34.05, -118.24), "la": (34.05, -118.24),
    "chicago": (41.88, -87.63), "houston": (29.76, -95.37),
    "phoenix": (33.45, -112.07), "philadelphia": (39.95, -75.17),
    "san antonio": (29.42, -98.49), "san diego": (32.72, -117.16),
    "dallas": (32.78, -96.80), "austin": (30.27, -97.74),
    "miami": (25.76, -80.19), "seattle": (47.61, -122.33),
    "denver": (39.74, -104.99), "boston": (42.36, -71.06),
    "atlanta": (33.75, -84.39), "portland": (45.52, -122.68),
    "las vegas": (36.17, -115.14), "detroit": (42.33, -83.05),
    "minneapolis": (44.98, -93.27), "san francisco": (37.77, -122.42),
    "london": (51.51, -0.13), "paris": (48.86, 2.35),
    "tokyo": (35.68, 139.69), "sydney": (-33.87, 151.21),
    "toronto": (43.65, -79.38), "berlin": (52.52, 13.41),
    "amsterdam": (52.37, 4.90), "singapore": (1.35, 103.82),
    "hong kong": (22.32, 114.17), "dubai": (25.20, 55.27),
    "mumbai": (19.08, 72.88), "seoul": (37.57, 126.98),
    "istanbul": (41.01, 28.98), "rome": (41.90, 12.50),
    "madrid": (40.42, -3.70), "milan": (45.46, 9.19),
    "busan": (35.18, 129.08), "bangkok": (13.76, 100.50),
}


def _extract_city(question: str) -> Optional[tuple[str, float, float]]:
    """Extract city name and coordinates from a market question."""
    q = question.lower()
    for city, (lat, lon) in CITY_COORDS.items():
        if city in q:
            return (city, lat, lon)
    return None


def _extract_temp_threshold(question: str) -> Optional[tuple[float, str]]:
    """Extract temperature threshold and unit from question.

    Handles patterns like:
    - "above 85°F"
    - "between 68-69°F"
    - "be 32°C"
    - "hit 100°F"
    """
    q = question.lower()

    # "between X-Y°F/C"
    m = re.search(r'between\s+(\d+)[–-](\d+)\s*°?\s*([fc])', q)
    if m:
        low, high = float(m.group(1)), float(m.group(2))
        unit = m.group(3).upper()
        # For "between X-Y", we want P(temp in [X, Y])
        return ((low + high) / 2, unit)

    # "be X°F/C"
    m = re.search(r'be\s+(\d+)\s*°?\s*([fc])', q)
    if m:
        return (float(m.group(1)), m.group(2).upper())

    # "above/hit/reach X°F/C"
    m = re.search(r'(?:above|hit|reach|exceed)\s+(\d+)\s*°?\s*([fc])', q)
    if m:
        return (float(m.group(1)), m.group(2).upper())

    # Generic "X°F" or "X°C"
    m = re.search(r'(\d+)\s*°\s*([fc])', q)
    if m:
        return (float(m.group(1)), m.group(2).upper())

    return None


def _f_to_c(f: float) -> float:
    return (f - 32) * 5 / 9


async def get_forecast(lat: float, lon: float, days: int = 7) -> Optional[dict]:
    """Fetch ensemble weather forecast from Open-Meteo."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(BASE_URL, params={
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min",
                "timezone": "auto",
                "forecast_days": days,
            })
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as e:
        logger.error(f"Open-Meteo forecast failed: {e}")
        return None


async def estimate_temp_probability(question: str) -> Optional[tuple[float, float, str]]:
    """Estimate the probability of a temperature event from a market question.

    Returns (probability, confidence, reasoning) or None.
    """
    city_info = _extract_city(question)
    if not city_info:
        return None

    city, lat, lon = city_info
    threshold_info = _extract_temp_threshold(question)
    if not threshold_info:
        return None

    temp_val, unit = threshold_info

    # Convert to Celsius for API
    if unit == "F":
        temp_c = _f_to_c(temp_val)
    else:
        temp_c = temp_val

    forecast = await get_forecast(lat, lon)
    if not forecast:
        return None

    daily = forecast.get("daily", {})
    max_temps = daily.get("temperature_2m_max", [])
    min_temps = daily.get("temperature_2m_min", [])
    dates = daily.get("time", [])

    if not max_temps:
        return None

    # Check if question mentions a specific date
    q_lower = question.lower()
    target_max = None

    for i, date in enumerate(dates):
        # Match "April 9" style
        if date in q_lower or _date_matches(date, q_lower):
            target_max = max_temps[i]
            break

    if target_max is None:
        # Use first forecast day
        target_max = max_temps[0]

    # Estimate probability
    # Simple model: forecast high vs threshold, with ~2°C uncertainty
    uncertainty = 2.0  # degrees C standard deviation
    diff = (temp_c - target_max) / uncertainty

    # Approximate normal CDF using logistic approximation
    # P(temp >= threshold) ≈ 1 / (1 + exp(1.7 * diff))
    import math
    prob_above = 1.0 / (1.0 + math.exp(1.7 * diff))

    # For "between X-Y" questions, probability is lower
    if "between" in q_lower:
        # Narrow band — typically ~10-20% chance for exact degree
        prob = min(0.30, prob_above * 0.4)
    elif "above" in q_lower or "hit" in q_lower or "exceed" in q_lower:
        prob = prob_above
    else:
        # "be X°C" — exact temperature, narrow band
        prob = min(0.30, prob_above * 0.4)

    prob = max(0.02, min(0.98, prob))
    confidence = 0.75  # Weather models are well-calibrated

    reasoning = f"Open-Meteo forecast: {city} high={target_max:.1f}°C, threshold={temp_c:.1f}°C ({temp_val}{unit})"
    logger.info(f"[WEATHER] {reasoning} -> prob={prob:.3f}")

    return (prob, confidence, reasoning)


def _date_matches(forecast_date: str, question: str) -> bool:
    """Check if a forecast date (YYYY-MM-DD) matches a date in the question."""
    from datetime import datetime
    try:
        dt = datetime.strptime(forecast_date, "%Y-%m-%d")
        # Check "April 9" format
        month_day = dt.strftime("%B %d").replace(" 0", " ").lower()
        if month_day in question:
            return True
        # Check "Apr 9" format
        month_day_short = dt.strftime("%b %d").replace(" 0", " ").lower()
        if month_day_short in question:
            return True
    except ValueError:
        pass
    return False
