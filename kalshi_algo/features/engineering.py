"""Turn raw :class:`MarketSnapshot` objects into a numeric feature matrix.

Design rules
------------
* **No look-ahead / no leakage.**  ``snapshot.result`` is never read here - it is
  the label, not a feature.  Only information observable at decision time is used.
* **Stable schema.**  The category vocabulary is learned at ``fit`` time so that
  train and test matrices always share columns (important for walk-forward CV).
* Features are deliberately aligned with the inefficiencies a prediction-market
  trader exploits: distance from the 0.5 "coin-flip" line (favorite-longshot),
  recent momentum and realized volatility, and liquidity proxies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.types import MarketSnapshot

# Numeric (non-category) features, in a fixed order.
NUMERIC_FEATURES = (
    "price",
    "dist_from_half",
    "price_sq",
    "spread",
    "log_volume",
    "log_open_interest",
    "oi_volume_ratio",
    "log_hours_to_close",
    "momentum",
    "momentum_recent",
    "volatility",
    "price_vs_hist_mean",
)

ALL_FEATURES = NUMERIC_FEATURES  # category dummies are appended dynamically


def _row(s: MarketSnapshot) -> dict[str, float]:
    price = float(np.clip(s.mid, 1e-3, 1 - 1e-3))
    hist = np.asarray(s.price_history, dtype=float) if s.price_history else np.array([price])
    momentum = float(hist[-1] - hist[0]) if hist.size > 1 else 0.0
    momentum_recent = float(hist[-1] - hist[-min(3, hist.size)]) if hist.size > 1 else 0.0
    volatility = float(np.std(hist)) if hist.size > 1 else 0.0
    return {
        "price": price,
        "dist_from_half": abs(price - 0.5),
        "price_sq": price * price,
        "spread": float(s.spread),
        "log_volume": float(np.log1p(s.volume)),
        "log_open_interest": float(np.log1p(s.open_interest)),
        "oi_volume_ratio": float(s.open_interest / (s.volume + 1.0)),
        "log_hours_to_close": float(np.log1p(s.hours_to_close)),
        "momentum": momentum,
        "momentum_recent": momentum_recent,
        "volatility": volatility,
        "price_vs_hist_mean": price - float(np.mean(hist)),
        "news_sentiment": float(s.extra.get("news_sentiment", 0.0)),
        "category": s.category,
    }


class FeatureBuilder:
    """Builds aligned feature matrices; learns the category vocabulary at fit."""

    def __init__(self, feature_set: tuple[str, ...] = ()):
        # If feature_set is empty we use the full numeric set.
        self.requested = tuple(feature_set) if feature_set else NUMERIC_FEATURES
        self.categories_: list[str] = []
        self.columns_: list[str] = []

    def fit(self, snapshots: list[MarketSnapshot]) -> "FeatureBuilder":
        cats = sorted({s.category for s in snapshots})
        self.categories_ = cats
        num_cols = [f for f in NUMERIC_FEATURES if f in self.requested]
        # Include the optional news-sentiment feature only when a provider has
        # actually attached it - keeps the offline pipeline byte-for-byte stable.
        if any("news_sentiment" in (s.extra or {}) for s in snapshots):
            num_cols.append("news_sentiment")
        self.columns_ = num_cols + [f"cat_{c}" for c in cats]
        return self

    def transform(self, snapshots: list[MarketSnapshot]) -> pd.DataFrame:
        if not self.columns_:
            raise RuntimeError("FeatureBuilder.transform called before fit")
        rows = [_row(s) for s in snapshots]
        df = pd.DataFrame(rows)
        # One-hot the category against the fitted vocabulary (unknown -> all 0).
        for c in self.categories_:
            df[f"cat_{c}"] = (df["category"] == c).astype(float)
        df = df.drop(columns=["category"])
        # Keep only fitted columns, in order; fill any gaps with 0.
        for col in self.columns_:
            if col not in df.columns:
                df[col] = 0.0
        return df[self.columns_].astype(float)

    def fit_transform(self, snapshots: list[MarketSnapshot]) -> pd.DataFrame:
        return self.fit(snapshots).transform(snapshots)

    @staticmethod
    def labels(snapshots: list[MarketSnapshot]) -> np.ndarray:
        return np.array([s.result for s in snapshots], dtype=float)
