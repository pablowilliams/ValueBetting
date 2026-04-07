"""
Per-category configuration for the value betting engine.
Maps market categories to their estimator pipelines and source weights.
"""

CATEGORY_CONFIG = {
    "sports": {
        "estimators": ["cross_market", "sports", "ai_ensemble"],
        "source_weights": {
            "cross_market": 0.30,
            "sports_odds": 0.45,   # Sportsbooks are the gold standard
            "ai_ensemble": 0.25,
        },
        "min_edge": 0.01,
        "min_sources": 1,
    },
    "weather": {
        "estimators": ["cross_market", "weather", "ai_ensemble"],
        "source_weights": {
            "cross_market": 0.25,
            "weather_model": 0.50,
            "ai_ensemble": 0.25,
        },
        "min_edge": 0.01,
        "min_sources": 1,
    },
    "politics": {
        "estimators": ["cross_market", "gdelt_news", "political", "ai_ensemble"],
        "source_weights": {
            "cross_market": 0.30,
            "gdelt_news": 0.25,
            "polling": 0.25,
            "ai_ensemble": 0.20,
        },
        "min_edge": 0.01,
        "min_sources": 1,
    },
    "crypto": {
        "estimators": ["cross_market", "crypto", "ai_ensemble"],
        "source_weights": {
            "cross_market": 0.30,
            "crypto_model": 0.35,
            "ai_ensemble": 0.20,
            "sentiment": 0.15,
        },
        "min_edge": 0.01,
        "min_sources": 1,
    },
    "economics": {
        "estimators": ["cross_market", "gdelt_news", "ai_ensemble"],
        "source_weights": {
            "cross_market": 0.35,
            "gdelt_news": 0.30,
            "ai_ensemble": 0.35,
        },
        "min_edge": 0.01,
        "min_sources": 1,
    },
    "other": {
        "estimators": ["cross_market", "gdelt_news", "ai_ensemble"],
        "source_weights": {
            "cross_market": 0.35,
            "gdelt_news": 0.30,
            "ai_ensemble": 0.35,
        },
        "min_edge": 0.01,
        "min_sources": 1,
    },
}


def get_category_config(category: str) -> dict:
    """Get config for a category, falling back to 'other'."""
    return CATEGORY_CONFIG.get(category, CATEGORY_CONFIG["other"])
