"""
Cross-Platform Event Matcher
Matches Polymarket markets to external data sources using fuzzy matching.
Generalized from SportsBetArb/matcher.py.
"""

import logging
import re
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Normalize text for matching — lowercase, strip punctuation."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def similarity(a: str, b: str) -> float:
    """String similarity ratio between two strings."""
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def extract_key_terms(question: str) -> list[str]:
    """Extract important terms from a market question for searching."""
    # Remove common question words
    stop_words = {
        "will", "the", "a", "an", "be", "is", "are", "was", "were", "do",
        "does", "did", "have", "has", "had", "can", "could", "would", "should",
        "may", "might", "shall", "of", "in", "on", "at", "to", "for", "with",
        "by", "from", "this", "that", "these", "those", "it", "its", "or",
        "and", "but", "if", "than", "before", "after", "above", "below",
        "yes", "no", "not",
    }

    text = normalize_text(question)
    words = text.split()

    # Keep words that are meaningful
    terms = [w for w in words if w not in stop_words and len(w) > 2]

    return terms


def match_score(poly_question: str, external_question: str) -> float:
    """Compute a match score between a Polymarket question and an external question.

    Returns 0-1 confidence score.
    """
    # Direct similarity
    direct_sim = similarity(poly_question, external_question)

    # Term overlap
    poly_terms = set(extract_key_terms(poly_question))
    ext_terms = set(extract_key_terms(external_question))

    if not poly_terms or not ext_terms:
        return direct_sim

    overlap = poly_terms & ext_terms
    jaccard = len(overlap) / len(poly_terms | ext_terms)

    # Weight: 40% direct similarity, 60% term overlap (more robust)
    score = 0.4 * direct_sim + 0.6 * jaccard

    return score


def is_good_match(poly_question: str, external_question: str, threshold: float = 0.35) -> bool:
    """Determine if two questions are about the same event."""
    return match_score(poly_question, external_question) >= threshold
