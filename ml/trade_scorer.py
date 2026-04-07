"""
Trade Quality Scorer — LightGBM model that predicts expected P&L per dollar.

The best ML approach for prediction market trading (based on research):
- LightGBM regression predicting expected P&L per dollar risked
- 28 features covering: trade setup, source agreement, market regime, category, history
- Warm-starts incrementally as new trades resolve
- Directly feeds Kelly sizing with learned parameters
- Falls back to heuristic scoring when <50 resolved trades

This model sits as a GATEKEEPER between signal generation and execution.
It consumes outputs from the calibration model and edge decay model as features.
"""

import json
import logging
import math
import os
import pickle
import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from config import settings

logger = logging.getLogger(__name__)

MODEL_PATH = "ml/trade_scorer_model.pkl"
TRAINING_DATA_PATH = "ml/trade_scorer_data.jsonl"
META_PATH = "ml/trade_scorer_meta.json"

CATEGORIES = ["sports", "weather", "politics", "crypto", "economics", "other"]
MIN_SAMPLES_FOR_ML = 50


@dataclass
class TradeFeatures:
    """Feature vector for the trade quality scorer."""
    # Trade setup
    entry_price: float
    edge_at_entry: float
    consensus_probability: float
    consensus_confidence: float
    bid_ask_spread: float
    orderbook_depth: float
    market_volume: float

    # Source agreement
    num_sources: int
    source_std: float
    source_range: float
    source_market_divergence: float  # abs(consensus - market_price)

    # Category
    category: str
    is_crypto: bool

    # Risk state
    heat_level: int       # 0=green, 1=yellow, 2=orange
    consecutive_losses: int
    daily_pnl: float
    bankroll_pct: float   # current bankroll / initial

    # Historical performance
    category_trade_count: int
    category_win_rate: float
    category_avg_pnl: float
    overall_win_rate: float
    overall_avg_pnl: float

    # Time
    hour_of_day: int

    # Sub-model outputs (meta-stacking)
    calibration_available: bool
    edge_decay_available: bool


@dataclass
class TradeOutcome:
    """Training sample: features + actual result."""
    features: dict       # TradeFeatures as dict
    actual_pnl: float    # P&L per dollar risked
    won: bool
    hold_seconds: float
    exit_reason: str
    timestamp: float


class TradeQualityScorer:
    """LightGBM model that predicts expected P&L per dollar risked.

    Usage:
        scorer = TradeQualityScorer()
        score = scorer.predict(features)  # returns expected P&L/dollar
        if score > 0.02:  # 2% expected return
            execute_trade()

        # After trade resolves:
        scorer.record_outcome(features, actual_pnl, won, hold_time, reason)
    """

    def __init__(self):
        self.model = None
        self._meta = {"n_samples": 0, "last_retrain": 0, "validation_mae": None}
        self._load()

    def _load(self):
        """Load saved model and metadata."""
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    self.model = pickle.load(f)
                logger.info("Loaded trade scorer model")
            except Exception as e:
                logger.warning(f"Failed to load trade scorer: {e}")

        if os.path.exists(META_PATH):
            try:
                with open(META_PATH) as f:
                    self._meta = json.load(f)
            except Exception:
                pass

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    @property
    def n_samples(self) -> int:
        return self._meta.get("n_samples", 0)

    def predict(self, features: TradeFeatures) -> tuple[float, float, str]:
        """Predict expected P&L per dollar risked.

        Returns:
            (expected_pnl, confidence, explanation)
        """
        if self.is_trained and self.n_samples >= MIN_SAMPLES_FOR_ML:
            return self._ml_predict(features)
        else:
            return self._heuristic_predict(features)

    def _heuristic_predict(self, f: TradeFeatures) -> tuple[float, float, str]:
        """Simple scoring formula for when ML has insufficient data."""
        score = (
            0.35 * f.edge_at_entry
            + 0.20 * f.consensus_confidence
            + 0.15 * max(0, 1.0 - f.source_std * 3)
            + 0.10 * min(1.0, f.num_sources / 3)
            + 0.10 * (1.0 if f.category_win_rate > 0.5 else 0.5)
            + 0.10 * (1.0 if f.heat_level == 0 else 0.5 if f.heat_level == 1 else 0.2)
        )
        confidence = 0.3 + 0.1 * f.num_sources
        reason = f"Heuristic: edge={f.edge_at_entry:.1%}, conf={f.consensus_confidence:.2f}, sources={f.num_sources}"
        return (score, confidence, reason)

    def _ml_predict(self, f: TradeFeatures) -> tuple[float, float, str]:
        """LightGBM prediction."""
        features_vec = self._build_feature_vector(f)
        features_2d = np.array([features_vec])
        predicted_pnl = float(self.model.predict(features_2d)[0])

        # Confidence from prediction magnitude
        confidence = min(0.95, 0.5 + abs(predicted_pnl) * 2)

        # Get top features for explanation
        importances = self.model.feature_importances_
        feature_names = self._feature_names()
        top = sorted(zip(feature_names, importances, features_vec),
                     key=lambda x: x[1], reverse=True)[:3]
        reason = f"ML score={predicted_pnl:.3f} | Top: {', '.join(f'{n}={v:.2f}' for n, _, v in top)}"

        return (predicted_pnl, confidence, reason)

    def _build_feature_vector(self, f: TradeFeatures) -> list[float]:
        """Convert TradeFeatures to numeric vector."""
        cat_idx = CATEGORIES.index(f.category) if f.category in CATEGORIES else 5

        return [
            f.entry_price,
            f.edge_at_entry,
            f.consensus_probability,
            f.consensus_confidence,
            f.bid_ask_spread,
            min(f.orderbook_depth, 10000) / 10000,  # normalize
            math.log1p(f.market_volume),
            f.num_sources,
            f.source_std,
            f.source_range,
            f.source_market_divergence,
            cat_idx,
            1.0 if f.is_crypto else 0.0,
            f.heat_level,
            f.consecutive_losses,
            f.daily_pnl,
            f.bankroll_pct,
            f.category_trade_count,
            f.category_win_rate,
            f.category_avg_pnl,
            f.overall_win_rate,
            f.overall_avg_pnl,
            math.sin(2 * math.pi * f.hour_of_day / 24),
            math.cos(2 * math.pi * f.hour_of_day / 24),
            1.0 if f.calibration_available else 0.0,
            1.0 if f.edge_decay_available else 0.0,
        ]

    def _feature_names(self) -> list[str]:
        return [
            "entry_price", "edge", "consensus_prob", "consensus_conf",
            "spread", "depth", "log_volume", "n_sources", "source_std",
            "source_range", "divergence", "category", "is_crypto",
            "heat", "consec_losses", "daily_pnl", "bankroll_pct",
            "cat_trade_count", "cat_win_rate", "cat_avg_pnl",
            "overall_wr", "overall_avg_pnl", "hour_sin", "hour_cos",
            "has_calibration", "has_edge_decay",
        ]

    def record_outcome(self, features: TradeFeatures, actual_pnl: float,
                       won: bool, hold_seconds: float, exit_reason: str):
        """Record a resolved trade for training."""
        outcome = TradeOutcome(
            features=asdict(features),
            actual_pnl=actual_pnl,
            won=won,
            hold_seconds=hold_seconds,
            exit_reason=exit_reason,
            timestamp=time.time(),
        )
        try:
            os.makedirs(os.path.dirname(TRAINING_DATA_PATH), exist_ok=True)
            with open(TRAINING_DATA_PATH, "a") as f:
                f.write(json.dumps(asdict(outcome)) + "\n")
            self._meta["n_samples"] = self._meta.get("n_samples", 0) + 1

            # Auto-retrain every 20 new samples
            if self._meta["n_samples"] >= MIN_SAMPLES_FOR_ML and self._meta["n_samples"] % 20 == 0:
                self.retrain()

        except IOError as e:
            logger.error(f"Failed to record outcome: {e}")

    def retrain(self) -> bool:
        """Train or warm-start the LightGBM model."""
        if not os.path.exists(TRAINING_DATA_PATH):
            return False

        samples = []
        with open(TRAINING_DATA_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

        if len(samples) < MIN_SAMPLES_FOR_ML:
            logger.info(f"Trade scorer: {len(samples)} samples < {MIN_SAMPLES_FOR_ML} needed")
            return False

        logger.info(f"Training trade scorer on {len(samples)} samples...")

        X, y = [], []
        for s in samples:
            feat = s["features"]
            tf = TradeFeatures(**feat)
            X.append(self._build_feature_vector(tf))
            y.append(s["actual_pnl"])

        X = np.array(X)
        y = np.array(y)

        try:
            import lightgbm as lgb

            # Determine hyperparams based on dataset size
            n = len(samples)
            params = {
                "objective": "regression",
                "metric": "mae",
                "n_estimators": 150 if n < 500 else 300,
                "max_depth": 4 if n < 500 else 6,
                "num_leaves": 15 if n < 500 else 31,
                "learning_rate": 0.05,
                "min_child_samples": 5 if n < 200 else 10,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "verbose": -1,
            }

            model = lgb.LGBMRegressor(**params)

            # Walk-forward split (70/30 time-ordered)
            split_idx = int(len(X) * 0.7)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # Warm-start from previous model if available
            init_model = MODEL_PATH if self.model is not None and os.path.exists(MODEL_PATH) else None

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.log_evaluation(period=0)],
                init_model=init_model,
            )

            # Validate
            val_pred = model.predict(X_val)
            val_mae = np.mean(np.abs(val_pred - y_val))

            # Safety check: don't deploy worse model
            if self.model is not None and self._meta.get("validation_mae"):
                if val_mae > self._meta["validation_mae"] * 1.2:
                    logger.warning(f"New model MAE {val_mae:.4f} > old {self._meta['validation_mae']:.4f} * 1.2, keeping old")
                    return False

            self.model = model
            self._meta["validation_mae"] = float(val_mae)
            self._meta["last_retrain"] = time.time()
            self._meta["n_samples"] = len(samples)

            # Save
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            with open(MODEL_PATH, "wb") as f:
                pickle.dump(model, f)
            with open(META_PATH, "w") as f:
                json.dump(self._meta, f)

            # Log feature importances
            importances = model.feature_importances_
            names = self._feature_names()
            top = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)[:5]
            logger.info(
                f"Trade scorer trained: MAE={val_mae:.4f}, "
                f"samples={len(samples)}, "
                f"top features: {', '.join(f'{n}={i:.3f}' for n, i in top)}"
            )
            return True

        except ImportError:
            logger.error("lightgbm not installed. pip install lightgbm")
            return False
        except Exception as e:
            logger.error(f"Trade scorer training failed: {e}")
            return False

    def get_stats(self) -> dict:
        """Get model statistics for dashboard."""
        return {
            "is_trained": self.is_trained,
            "n_samples": self.n_samples,
            "validation_mae": self._meta.get("validation_mae"),
            "last_retrain": self._meta.get("last_retrain", 0),
            "phase": "ML" if self.is_trained and self.n_samples >= MIN_SAMPLES_FOR_ML
                     else f"Heuristic ({self.n_samples}/{MIN_SAMPLES_FOR_ML} samples)",
        }


def build_trade_features(
    market,
    consensus,
    edge_signal,
    risk_manager,
    calibration_model=None,
    edge_decay_model=None,
    trade_logger=None,
) -> TradeFeatures:
    """Construct TradeFeatures from bot state for the scorer."""
    probs = [e.probability for e in consensus.estimates]

    # Historical stats
    cat_stats = {"trades": 0, "wins": 0, "pnl": 0.0}
    overall_stats = {"trades": 0, "wins": 0, "pnl": 0.0}

    if trade_logger:
        stats = trade_logger.get_stats()
        for c in stats.get("by_category", []):
            if c["category"] == market.category:
                cat_stats = c
        overall_stats["trades"] = stats.get("total_trades", 0) or 0
        overall_stats["wins"] = stats.get("wins", 0) or 0
        overall_stats["pnl"] = stats.get("total_pnl", 0) or 0

    cat_wr = cat_stats["wins"] / cat_stats["trades"] if cat_stats["trades"] > 0 else 0.5
    cat_avg = cat_stats["pnl"] / cat_stats["trades"] if cat_stats["trades"] > 0 else 0.0
    overall_wr = overall_stats["wins"] / overall_stats["trades"] if overall_stats["trades"] > 0 else 0.5
    overall_avg = overall_stats["pnl"] / overall_stats["trades"] if overall_stats["trades"] > 0 else 0.0

    heat_map = {"GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3}

    return TradeFeatures(
        entry_price=edge_signal.entry_price,
        edge_at_entry=edge_signal.net_edge,
        consensus_probability=consensus.probability,
        consensus_confidence=consensus.confidence,
        bid_ask_spread=market.spread,
        orderbook_depth=market.orderbook_depth,
        market_volume=market.volume,
        num_sources=consensus.sources,
        source_std=float(np.std(probs)) if len(probs) > 1 else 0.5,
        source_range=max(probs) - min(probs) if len(probs) > 1 else 0.0,
        source_market_divergence=abs(consensus.probability - market.yes_price),
        category=market.category,
        is_crypto=market.category == "crypto",
        heat_level=heat_map.get(risk_manager.heat_level, 0),
        consecutive_losses=risk_manager.consecutive_losses,
        daily_pnl=risk_manager.daily_pnl,
        bankroll_pct=risk_manager.bankroll / risk_manager.initial_bankroll,
        category_trade_count=cat_stats.get("trades", 0),
        category_win_rate=cat_wr,
        category_avg_pnl=cat_avg,
        overall_win_rate=overall_wr,
        overall_avg_pnl=overall_avg,
        hour_of_day=time.localtime().tm_hour,
        calibration_available=calibration_model is not None and calibration_model.is_trained,
        edge_decay_available=edge_decay_model is not None and edge_decay_model.is_trained,
    )
