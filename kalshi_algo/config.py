"""Central configuration objects.

Everything that the learning loop is allowed to tune lives in :class:`StrategyConfig`
so that hyperparameter search has a single, well-typed surface to optimize over.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from typing import Any


# --- Kalshi exchange constants -------------------------------------------------

# Kalshi charges a trading fee per contract.  As of the public fee schedule the
# general fee is round_up(0.07 * C * P * (1 - P)) dollars for C contracts at
# price P (in dollars, 0..1).  Settlement pays $1 for a winning contract.
KALSHI_FEE_COEFFICIENT = 0.07
CONTRACT_PAYOUT = 1.0

# Categories we explicitly exclude.  The mandate is "not weather".
EXCLUDED_CATEGORIES = ("weather", "climate", "temperature")


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for the gradient-boosted fair-value predictor."""

    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 3
    min_samples_leaf: int = 40
    subsample: float = 0.8
    max_features: float = 0.8
    calibration: str = "isotonic"  # one of {"isotonic", "sigmoid", "none"}
    random_state: int = 7


@dataclass(frozen=True)
class StrategyConfig:
    """Everything the optimizer is allowed to tune.

    Splitting model vs. trading parameters lets the loop search them jointly.
    """

    model: ModelConfig = field(default_factory=ModelConfig)

    # Trading rules ---------------------------------------------------------
    edge_threshold: float = 0.04      # min |model_prob - price| to act
    kelly_fraction: float = 0.10      # fraction of full Kelly used for sizing
    max_position_frac: float = 0.015  # cap any single position to this % of bankroll
    min_price: float = 0.05           # avoid extreme longshots / near-certain
    max_price: float = 0.95
    min_volume: float = 50.0          # liquidity filter (contracts traded)
    max_spread: float = 0.06          # skip illiquid / wide markets
    allow_short_no: bool = True       # also trade the NO side when edge is negative

    # Feature toggles (the loop can prune noisy features) -------------------
    feature_set: tuple[str, ...] = ()  # empty => use the full default set

    def with_updates(self, **kwargs: Any) -> "StrategyConfig":
        """Return a copy with top-level fields replaced."""
        return replace(self, **kwargs)

    def with_model(self, **kwargs: Any) -> "StrategyConfig":
        """Return a copy with model sub-fields replaced."""
        return replace(self, model=replace(self.model, **kwargs))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PerformanceTargets:
    """Acceptance criteria for the self-improving loop.

    The loop keeps iterating until an out-of-sample configuration clears *all*
    of these on walk-forward validation (or the iteration budget is exhausted).
    """

    min_sharpe: float = 1.0           # annualized-ish per-trade Sharpe
    min_roi: float = 0.05             # return on capital deployed
    max_drawdown: float = 0.25        # worst peak-to-trough on equity curve
    min_hit_rate: float = 0.50        # fraction of profitable trades
    min_trades: int = 200             # statistical-significance floor
    min_profit_factor: float = 1.15   # gross win / gross loss
    max_brier: float = 0.25           # predictor must beat a coin flip (0.25)

    def is_met(self, m: dict[str, float]) -> bool:
        return (
            m.get("sharpe", -1e9) >= self.min_sharpe
            and m.get("roi", -1e9) >= self.min_roi
            and m.get("max_drawdown", 1e9) <= self.max_drawdown
            and m.get("hit_rate", 0.0) >= self.min_hit_rate
            and m.get("n_trades", 0) >= self.min_trades
            and m.get("profit_factor", 0.0) >= self.min_profit_factor
            and m.get("brier", 1e9) <= self.max_brier
        )
