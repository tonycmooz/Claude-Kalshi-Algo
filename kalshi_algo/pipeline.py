"""Convenience factory: fit a complete :class:`Strategy` from training data."""
from __future__ import annotations

from .config import StrategyConfig
from .data.types import MarketSnapshot
from .features import FeatureBuilder
from .models import FairValuePredictor
from .strategy import Strategy


def train_strategy(train: list[MarketSnapshot], config: StrategyConfig) -> Strategy:
    """Fit the feature builder + predictor on ``train`` and return a Strategy.

    Only resolved markets (``result is not None``) can be used as labels.
    """
    labelled = [s for s in train if s.result is not None]
    fb = FeatureBuilder(config.feature_set).fit(labelled)
    X = fb.transform(labelled).values
    y = FeatureBuilder.labels(labelled)
    predictor = FairValuePredictor(config.model).fit(X, y)
    return Strategy(config, predictor, fb)
