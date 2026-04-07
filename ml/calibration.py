"""
ML Probability Calibration Model
----------------------------------
XGBoost model that learns optimal source weighting from resolved trades.

Instead of fixed category weights (e.g., sportsbooks=0.45, AI=0.25),
this model learns which sources are most predictive for each category
from actual trade outcomes (Brier score optimization).

Features per trade:
  - Each source's probability estimate
  - Each source's confidence
  - Number of sources available
  - Category (one-hot encoded)
  - Market price at time of estimate
  - Source agreement (std dev of estimates)

Target:
  - Actual outcome (1 = YES resolved, 0 = NO resolved)

Output:
  - Calibrated probability that is better than fixed-weight consensus
"""

import json
import logging
import os
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.market import ProbEstimate

logger = logging.getLogger(__name__)

MODEL_PATH = "ml/calibration_model.pkl"
TRAINING_DATA_PATH = "ml/calibration_data.jsonl"

# All possible source names — used for consistent feature ordering
SOURCE_NAMES = [
    "cross_market", "sports_odds", "weather_model", "polling",
    "crypto_model", "sentiment", "ai_ensemble",
]

CATEGORIES = ["sports", "weather", "politics", "crypto", "economics", "other"]


@dataclass
class CalibrationSample:
    """One training sample for the calibration model."""
    source_probs: dict[str, float]     # source_name -> probability
    source_confs: dict[str, float]     # source_name -> confidence
    category: str
    market_price: float
    num_sources: int
    actual_outcome: int                # 1=YES, 0=NO


class CalibrationModel:
    """XGBoost-based probability calibration.

    Learns to combine source estimates better than fixed weights
    by training on resolved trade outcomes.
    """

    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load saved model if it exists."""
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    self.model = pickle.load(f)
                logger.info("Loaded calibration model from disk")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
                self.model = None

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    def predict(
        self,
        estimates: list[ProbEstimate],
        category: str,
        market_price: float = 0.5,
    ) -> tuple[float, float]:
        """Predict calibrated probability from source estimates.

        Returns:
            (calibrated_probability, confidence)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")

        features = self._build_features(estimates, category, market_price)
        features_2d = np.array([features])

        prob = float(self.model.predict_proba(features_2d)[0, 1])
        # Confidence based on model certainty (distance from 0.5)
        confidence = 0.5 + abs(prob - 0.5)

        return (prob, confidence)

    def _build_features(
        self,
        estimates: list[ProbEstimate],
        category: str,
        market_price: float,
    ) -> list[float]:
        """Build feature vector from estimates."""
        # Source probabilities (0 if not available)
        source_probs = {e.source: e.probability for e in estimates}
        source_confs = {e.source: e.confidence for e in estimates}

        features = []

        # Source probability features
        for name in SOURCE_NAMES:
            features.append(source_probs.get(name, 0.0))

        # Source confidence features
        for name in SOURCE_NAMES:
            features.append(source_confs.get(name, 0.0))

        # Source availability (binary)
        for name in SOURCE_NAMES:
            features.append(1.0 if name in source_probs else 0.0)

        # Category one-hot
        for cat in CATEGORIES:
            features.append(1.0 if category == cat else 0.0)

        # Meta features
        probs = [e.probability for e in estimates]
        features.append(market_price)
        features.append(len(estimates))
        features.append(np.std(probs) if len(probs) > 1 else 0.5)  # Source disagreement
        features.append(np.mean(probs) if probs else 0.5)           # Mean estimate
        features.append(max(probs) - min(probs) if len(probs) > 1 else 0.0)  # Range

        return features

    def save_training_sample(self, sample: CalibrationSample):
        """Append a training sample to the data file."""
        record = {
            "source_probs": sample.source_probs,
            "source_confs": sample.source_confs,
            "category": sample.category,
            "market_price": sample.market_price,
            "num_sources": sample.num_sources,
            "actual_outcome": sample.actual_outcome,
        }
        try:
            os.makedirs(os.path.dirname(TRAINING_DATA_PATH), exist_ok=True)
            with open(TRAINING_DATA_PATH, "a") as f:
                f.write(json.dumps(record) + "\n")
        except IOError as e:
            logger.error(f"Failed to save training sample: {e}")

    def train(self, min_samples: int = 50) -> bool:
        """Train or retrain the model from saved data.

        Returns True if training succeeded.
        """
        if not os.path.exists(TRAINING_DATA_PATH):
            logger.info("No training data available")
            return False

        # Load training data
        samples = []
        with open(TRAINING_DATA_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

        if len(samples) < min_samples:
            logger.info(f"Not enough data: {len(samples)} < {min_samples}")
            return False

        logger.info(f"Training calibration model on {len(samples)} samples...")

        # Build feature matrix
        X = []
        y = []
        for s in samples:
            # Reconstruct ProbEstimate-like objects for feature building
            estimates = []
            for source, prob in s["source_probs"].items():
                estimates.append(ProbEstimate(
                    probability=prob,
                    confidence=s["source_confs"].get(source, 0.5),
                    source=source,
                ))
            features = self._build_features(
                estimates, s["category"], s["market_price"]
            )
            X.append(features)
            y.append(s["actual_outcome"])

        X = np.array(X)
        y = np.array(y)

        try:
            from sklearn.model_selection import cross_val_score
            from xgboost import XGBClassifier

            model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
            )

            # Cross-validate
            scores = cross_val_score(model, X, y, cv=5, scoring="neg_brier_score")
            avg_brier = -np.mean(scores)
            logger.info(f"Cross-val Brier score: {avg_brier:.4f}")

            # Train on full data
            model.fit(X, y)
            self.model = model

            # Save
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            with open(MODEL_PATH, "wb") as f:
                pickle.dump(model, f)

            # Log feature importances
            importances = model.feature_importances_
            feature_names = (
                [f"{s}_prob" for s in SOURCE_NAMES]
                + [f"{s}_conf" for s in SOURCE_NAMES]
                + [f"{s}_avail" for s in SOURCE_NAMES]
                + [f"cat_{c}" for c in CATEGORIES]
                + ["market_price", "num_sources", "source_disagreement",
                   "mean_estimate", "estimate_range"]
            )
            top_features = sorted(
                zip(feature_names, importances), key=lambda x: x[1], reverse=True
            )[:10]
            logger.info("Top features: " + ", ".join(
                f"{name}={imp:.3f}" for name, imp in top_features
            ))

            return True

        except ImportError:
            logger.error("xgboost or sklearn not installed. pip install xgboost scikit-learn")
            return False
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False

    def get_source_brier_scores(self) -> dict[str, float]:
        """Compute per-source Brier scores from training data.

        Lower Brier = more calibrated. Used for dashboard display.
        """
        if not os.path.exists(TRAINING_DATA_PATH):
            return {}

        samples = []
        with open(TRAINING_DATA_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

        if not samples:
            return {}

        brier_by_source: dict[str, list[float]] = {}
        for s in samples:
            outcome = s["actual_outcome"]
            for source, prob in s["source_probs"].items():
                if source not in brier_by_source:
                    brier_by_source[source] = []
                brier_by_source[source].append((prob - outcome) ** 2)

        return {
            source: sum(scores) / len(scores)
            for source, scores in brier_by_source.items()
            if scores
        }
