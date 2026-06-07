"""Search space, sampling/mutation, and the fitness function.

The optimizer treats a :class:`StrategyConfig` as a point in a mixed
continuous/discrete space and searches it to maximize a single fitness scalar
derived from **walk-forward (out-of-sample)** performance.  Crucially the
fitness rewards *stability across folds*, not just the average, so the loop
cannot win by overfitting one lucky period.
"""
from __future__ import annotations

import numpy as np

from ..config import ModelConfig, StrategyConfig
from ..backtest.walkforward import WalkForwardResult


def objective_score(wf: WalkForwardResult) -> float:
    """Scalar fitness — higher is better.

    Blends risk-adjusted return (Sharpe), capital efficiency (ROI) and
    cross-fold robustness, while penalizing drawdown and poor calibration.
    """
    a = wf.aggregate
    s = wf.stability
    if a["n_trades"] < 30:
        return -10.0 + a["n_trades"] / 30.0  # too few trades to trust
    return (
        0.6 * a["sharpe"]
        + 10.0 * a["roi"]
        + 3.0 * s.get("frac_profitable_folds", 0.0)
        + 1.0 * s.get("worst_fold_sharpe", 0.0)
        - 5.0 * a["max_drawdown"]
        - 8.0 * max(0.0, a["brier"] - 0.20)
        - 1.0 * s.get("sharpe_std", 0.0)
    )


# --- search space --------------------------------------------------------------

def sample_config(rng: np.random.Generator) -> StrategyConfig:
    """Draw a fresh random configuration from sensible ranges."""
    model = ModelConfig(
        n_estimators=int(rng.choice([150, 200, 300, 400])),
        learning_rate=float(np.exp(rng.uniform(np.log(0.02), np.log(0.12)))),
        max_depth=int(rng.integers(2, 5)),
        min_samples_leaf=int(rng.choice([20, 40, 60, 90, 120])),
        subsample=float(rng.uniform(0.6, 1.0)),
        max_features=float(rng.uniform(0.5, 1.0)),
        calibration=str(rng.choice(["isotonic", "sigmoid"])),
    )
    return StrategyConfig(
        model=model,
        edge_threshold=float(rng.uniform(0.03, 0.10)),
        kelly_fraction=float(rng.uniform(0.04, 0.22)),
        max_position_frac=float(rng.uniform(0.005, 0.030)),
        min_price=float(rng.uniform(0.03, 0.10)),
        max_price=float(rng.uniform(0.90, 0.97)),
        max_spread=float(rng.uniform(0.03, 0.08)),
        min_volume=float(rng.choice([20, 50, 80, 120])),
        allow_short_no=True,
    )


def _jitter(value: float, scale: float, lo: float, hi: float,
            rng: np.random.Generator) -> float:
    span = hi - lo
    return float(np.clip(value + rng.normal(0.0, scale * span), lo, hi))


def mutate_config(base: StrategyConfig, rng: np.random.Generator,
                  scale: float = 0.25) -> StrategyConfig:
    """Local perturbation of a good config (exploitation around the incumbent)."""
    m = base.model
    model = ModelConfig(
        n_estimators=int(np.clip(
            m.n_estimators + rng.choice([-100, -50, 0, 50, 100]), 100, 600)),
        learning_rate=_jitter(m.learning_rate, scale, 0.02, 0.12, rng),
        max_depth=int(np.clip(m.max_depth + rng.choice([-1, 0, 0, 1]), 2, 5)),
        min_samples_leaf=int(np.clip(
            m.min_samples_leaf + rng.choice([-20, 0, 20]), 20, 150)),
        subsample=_jitter(m.subsample, scale, 0.6, 1.0, rng),
        max_features=_jitter(m.max_features, scale, 0.5, 1.0, rng),
        calibration=(m.calibration if rng.random() > 0.2
                     else str(rng.choice(["isotonic", "sigmoid"]))),
    )
    return base.with_updates(
        model=model,
        edge_threshold=_jitter(base.edge_threshold, scale, 0.03, 0.12, rng),
        kelly_fraction=_jitter(base.kelly_fraction, scale, 0.04, 0.25, rng),
        max_position_frac=_jitter(base.max_position_frac, scale, 0.005, 0.030, rng),
        max_spread=_jitter(base.max_spread, scale, 0.03, 0.08, rng),
    )
