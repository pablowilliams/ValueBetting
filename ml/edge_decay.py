"""
ML Edge Decay Predictor
-------------------------
Predicts how quickly an edge will close (or grow), optimizing entry/exit timing.

Instead of fixed exit rules (exit at 1% edge), this model learns:
- How fast edges typically decay for each category
- Whether current conditions suggest holding longer or exiting early
- Whether the edge is likely to grow (momentum) or shrink (reversion)

Features:
  - Current edge, entry edge, edge change rate
  - Time held (seconds)
  - Category
  - Market volume and liquidity
  - Number of sources at entry
  - Source agreement at entry
  - P&L so far

Target:
  - Optimal action: HOLD (0), EXIT_PROFIT (1), EXIT_LOSS (2)
  - Derived from hindsight: what action would have maximized P&L

This model is trained from completed trades using the full price history.
"""

import json
import logging
import os
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_PATH = "ml/edge_decay_model.pkl"
TRAINING_DATA_PATH = "ml/edge_decay_data.jsonl"

CATEGORIES = ["sports", "weather", "politics", "crypto", "economics", "other"]


@dataclass
class EdgeDecaySample:
    """One training sample for the edge decay model."""
    entry_edge: float
    current_edge: float
    edge_change_rate: float   # Edge change per minute
    pnl: float
    age_seconds: float
    category: str
    entry_price: float
    current_price: float
    volume: float
    num_sources: int
    source_agreement: float   # 1 - std_dev of source estimates
    action_taken: int         # 0=HOLD, 1=EXIT_PROFIT, 2=EXIT_LOSS
    final_pnl: float          # What the P&L ended up being


class EdgeDecayModel:
    """Predicts optimal exit timing based on edge decay patterns."""

    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    self.model = pickle.load(f)
                logger.info("Loaded edge decay model from disk")
            except Exception as e:
                logger.warning(f"Failed to load edge decay model: {e}")
                self.model = None

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    def predict(
        self,
        edge: float,
        pnl: float,
        age_seconds: float,
        category: str,
        entry_price: float,
        current_price: float,
        volume: float = 0,
        num_sources: int = 2,
        source_agreement: float = 0.7,
        entry_edge: float = None,
    ) -> tuple[bool, float, str]:
        """Predict whether to exit now.

        Returns:
            (should_exit, confidence, reason)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")

        if entry_edge is None:
            entry_edge = edge + (pnl if pnl > 0 else 0)

        # Edge change rate (per minute)
        age_minutes = max(age_seconds / 60, 0.1)
        edge_change_rate = (edge - entry_edge) / age_minutes

        features = self._build_features(
            entry_edge=entry_edge,
            current_edge=edge,
            edge_change_rate=edge_change_rate,
            pnl=pnl,
            age_seconds=age_seconds,
            category=category,
            entry_price=entry_price,
            current_price=current_price,
            volume=volume,
            num_sources=num_sources,
            source_agreement=source_agreement,
        )

        features_2d = np.array([features])
        pred = self.model.predict(features_2d)[0]
        proba = self.model.predict_proba(features_2d)[0]

        confidence = float(max(proba))

        if pred == 0:
            return (False, confidence, "ML: HOLD — edge likely to persist")
        elif pred == 1:
            return (True, confidence, f"ML: EXIT_PROFIT — edge decaying (rate={edge_change_rate:.4f}/min)")
        else:
            return (True, confidence, f"ML: EXIT_LOSS — edge unlikely to recover")

    def _build_features(self, **kwargs) -> list[float]:
        """Build feature vector."""
        features = [
            kwargs["entry_edge"],
            kwargs["current_edge"],
            kwargs["edge_change_rate"],
            kwargs["pnl"],
            kwargs["age_seconds"],
            kwargs["age_seconds"] / 3600,       # Hours held
            kwargs["entry_price"],
            kwargs["current_price"],
            kwargs["current_price"] - kwargs["entry_price"],  # Price change
            kwargs["volume"],
            kwargs["num_sources"],
            kwargs["source_agreement"],
            kwargs["current_edge"] / max(kwargs["entry_edge"], 0.001),  # Edge ratio
        ]

        # Category one-hot
        for cat in CATEGORIES:
            features.append(1.0 if kwargs["category"] == cat else 0.0)

        return features

    def save_training_sample(self, sample: EdgeDecaySample):
        """Append training sample."""
        record = {
            "entry_edge": sample.entry_edge,
            "current_edge": sample.current_edge,
            "edge_change_rate": sample.edge_change_rate,
            "pnl": sample.pnl,
            "age_seconds": sample.age_seconds,
            "category": sample.category,
            "entry_price": sample.entry_price,
            "current_price": sample.current_price,
            "volume": sample.volume,
            "num_sources": sample.num_sources,
            "source_agreement": sample.source_agreement,
            "action_taken": sample.action_taken,
            "final_pnl": sample.final_pnl,
        }
        try:
            os.makedirs(os.path.dirname(TRAINING_DATA_PATH), exist_ok=True)
            with open(TRAINING_DATA_PATH, "a") as f:
                f.write(json.dumps(record) + "\n")
        except IOError as e:
            logger.error(f"Failed to save edge decay sample: {e}")

    def train(self, min_samples: int = 30) -> bool:
        """Train the edge decay model from saved data."""
        if not os.path.exists(TRAINING_DATA_PATH):
            logger.info("No edge decay training data available")
            return False

        samples = []
        with open(TRAINING_DATA_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

        if len(samples) < min_samples:
            logger.info(f"Not enough data: {len(samples)} < {min_samples}")
            return False

        logger.info(f"Training edge decay model on {len(samples)} samples...")

        X = []
        y = []
        for s in samples:
            features = self._build_features(**{k: s[k] for k in [
                "entry_edge", "current_edge", "edge_change_rate", "pnl",
                "age_seconds", "category", "entry_price", "current_price",
                "volume", "num_sources", "source_agreement",
            ]})
            X.append(features)
            y.append(s["action_taken"])

        X = np.array(X)
        y = np.array(y)

        try:
            from xgboost import XGBClassifier

            model = XGBClassifier(
                n_estimators=80,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                random_state=42,
            )

            model.fit(X, y)
            self.model = model

            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            with open(MODEL_PATH, "wb") as f:
                pickle.dump(model, f)

            logger.info("Edge decay model trained and saved")
            return True

        except ImportError:
            logger.error("xgboost not installed. pip install xgboost")
            return False
        except Exception as e:
            logger.error(f"Edge decay training failed: {e}")
            return False
