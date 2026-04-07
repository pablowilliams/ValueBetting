"""
Consensus Engine — Weighted aggregation of probability estimates.
The core intelligence of the value betting system.

Combines estimates from multiple sources using category-specific weights,
accounting for source agreement and confidence.
"""

import logging

from config.categories import get_category_config
from core.market import ProbEstimate, ConsensusEstimate

logger = logging.getLogger(__name__)


def compute_consensus(
    estimates: list[ProbEstimate],
    category: str,
    ml_model=None,
) -> ConsensusEstimate:
    """Compute weighted consensus from multiple probability estimates.

    If an ML calibration model is provided, it overrides fixed weights
    with learned optimal weighting.

    Args:
        estimates: List of probability estimates from different sources
        category: Market category for weight lookup
        ml_model: Optional trained ML calibration model

    Returns:
        ConsensusEstimate with probability, confidence, and source info
    """
    if not estimates:
        return ConsensusEstimate(probability=0.5, confidence=0.0, sources=0)

    # ── ML Path: Use trained calibration model if available ──
    if ml_model is not None:
        try:
            ml_prob, ml_conf = ml_model.predict(estimates, category)
            logger.info(
                f"ML consensus: prob={ml_prob:.3f} conf={ml_conf:.2f} "
                f"from {len(estimates)} sources [{category}]"
            )
            return ConsensusEstimate(
                probability=ml_prob,
                confidence=ml_conf,
                sources=len(estimates),
                estimates=estimates,
            )
        except Exception as e:
            logger.warning(f"ML model failed, falling back to fixed weights: {e}")

    # ── Fixed Weight Path ──
    cat_config = get_category_config(category)
    source_weights = cat_config["source_weights"]

    weighted_sum = 0.0
    weight_total = 0.0

    for est in estimates:
        base_weight = source_weights.get(est.source, 0.10)
        # Scale by individual source confidence
        w = base_weight * est.confidence
        weighted_sum += w * est.probability
        weight_total += w

    if weight_total == 0:
        return ConsensusEstimate(probability=0.5, confidence=0.0, sources=0)

    mean_prob = weighted_sum / weight_total

    # ── Confidence Scoring ──
    # Based on: source count, agreement between sources, individual confidences
    probs = [e.probability for e in estimates]

    # Agreement: how close are the estimates to each other?
    if len(probs) > 1:
        disagreement = max(probs) - min(probs)
        agreement_score = max(0.0, 1.0 - disagreement * 2)
    else:
        agreement_score = 0.7  # Single source = reasonable confidence

    # Source count: more sources = more confidence (caps at 3)
    source_score = max(0.5, min(1.0, len(estimates) / 3))

    # Average individual confidence
    avg_conf = sum(e.confidence for e in estimates) / len(estimates)

    confidence = agreement_score * source_score * avg_conf
    confidence = max(0.15, min(1.0, confidence))

    logger.info(
        f"Consensus [{category}]: prob={mean_prob:.3f} conf={confidence:.2f} | "
        f"sources={len(estimates)} agreement={agreement_score:.2f} | "
        f"probs={[f'{p:.2f}' for p in probs]}"
    )

    return ConsensusEstimate(
        probability=mean_prob,
        confidence=confidence,
        sources=len(estimates),
        estimates=estimates,
    )
