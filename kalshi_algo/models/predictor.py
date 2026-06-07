"""The fair-value predictor.

Estimates ``P(market resolves YES)`` from the engineered features.  The whole
strategy hinges on this estimate being *better calibrated* than the market's own
implied probability (the price); the gap is the tradeable edge.

A gradient-boosted tree ensemble captures the non-linear interactions between
price, distance-from-half, momentum and liquidity that encode the structural
mispricings.  Predicted scores are then **probability-calibrated** on a held-out
slice (isotonic or Platt) so that "0.62" really means a 62% chance - essential
because position sizing (Kelly) is extremely sensitive to probability accuracy.
"""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier

from ..config import ModelConfig


def _calibrate_prefit(base, X_cal, y_cal, method: str):
    """Calibrate an already-fitted estimator on a held-out slice.

    sklearn >=1.6 replaced ``cv="prefit"`` with wrapping the fitted model in a
    ``FrozenEstimator``; we support both so the code runs across versions.
    """
    try:
        from sklearn.frozen import FrozenEstimator
        cal = CalibratedClassifierCV(FrozenEstimator(base), method=method)
        return cal.fit(X_cal, y_cal)
    except ImportError:  # older sklearn
        cal = CalibratedClassifierCV(base, method=method, cv="prefit")
        return cal.fit(X_cal, y_cal)


class FairValuePredictor:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None  # fitted estimator (calibrated or raw)

    def _make_base(self) -> GradientBoostingClassifier:
        c = self.config
        return GradientBoostingClassifier(
            n_estimators=c.n_estimators,
            learning_rate=c.learning_rate,
            max_depth=c.max_depth,
            min_samples_leaf=c.min_samples_leaf,
            subsample=c.subsample,
            max_features=c.max_features,
            random_state=c.random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FairValuePredictor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        base = self._make_base()

        if self.config.calibration in ("isotonic", "sigmoid"):
            # Calibrate on a held-out tail slice (time-ordered, so still
            # respects causality) to avoid the 3x cost of internal CV.
            n = len(y)
            cut = max(int(n * 0.8), n - 1500)
            cut = min(cut, n - 50) if n > 100 else max(1, n - 1)
            Xtr, ytr = X[:cut], y[:cut]
            Xcal, ycal = X[cut:], y[cut:]
            if len(np.unique(ycal)) < 2 or len(ycal) < 30:
                # Not enough calibration data -> fall back to raw fit.
                self.model = base.fit(X, y)
            else:
                base.fit(Xtr, ytr)
                self.model = _calibrate_prefit(
                    base, Xcal, ycal, self.config.calibration)
        else:
            self.model = base.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(YES) for each row."""
        if self.model is None:
            raise RuntimeError("predict_proba called before fit")
        X = np.asarray(X, dtype=float)
        p = self.model.predict_proba(X)
        # Column for class "1" (YES).
        classes = list(self.model.classes_)
        idx = classes.index(1) if 1 in classes else (len(classes) - 1)
        return np.clip(p[:, idx], 1e-4, 1 - 1e-4)
