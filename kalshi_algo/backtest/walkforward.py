"""Walk-forward (out-of-sample) validation.

This is the guard against the cardinal sin of trading research - overfitting.
The data is split into chronological blocks; for each step the model is trained
on everything *before* a block and evaluated on the block itself.  A short
**embargo** gap between train and test removes any boundary leakage.

Two views are produced:

* ``aggregate`` - metrics over *all* out-of-sample trades stitched into a single
  equity curve (the headline, statistically robust numbers); and
* ``folds``     - per-fold metrics, from which we derive *stability* statistics
  (worst-fold Sharpe, fraction of profitable folds).  A strategy that is only
  good on average but loses in half its folds is correctly penalized.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import StrategyConfig
from ..data.types import MarketSnapshot
from ..pipeline import train_strategy
from .engine import Backtester
from .metrics import compute_metrics


@dataclass
class WalkForwardResult:
    aggregate: dict[str, float]
    folds: list[dict[str, float]] = field(default_factory=list)
    stability: dict[str, float] = field(default_factory=dict)
    equity_curve: list[float] = field(default_factory=list)


def walk_forward_validate(
    snapshots: list[MarketSnapshot],
    config: StrategyConfig,
    n_folds: int = 5,
    embargo: int = 50,
    initial_bankroll: float = 10_000.0,
) -> WalkForwardResult:
    snaps = sorted(snapshots, key=lambda s: s.observed_ts)
    n = len(snaps)
    # n_folds test blocks; the first block is reserved entirely for training.
    block = n // (n_folds + 1)
    if block < 50:
        raise ValueError("Not enough data for the requested number of folds")

    all_returns, all_pnl, all_deployed, all_wins = [], [], [], []
    all_briers, fold_metrics = [], []
    bankroll = initial_bankroll
    equity = [bankroll]

    for k in range(1, n_folds + 1):
        train_end = k * block - embargo
        test_start = k * block
        test_end = (k + 1) * block if k < n_folds else n
        train = snaps[:max(train_end, block)]
        test = snaps[test_start:test_end]
        if len(test) < 20 or not any(s.result is not None for s in train):
            continue

        strategy = train_strategy(train, config)
        # Run the fold starting from the *current* compounded bankroll so the
        # stitched equity curve is continuous and Kelly sizing carries across.
        bt = Backtester(config, initial_bankroll=bankroll)
        res = bt.run(strategy, test)
        fold_metrics.append({"fold": k, **res.metrics})

        for t in res.trades:
            q = t["price"]
            cost = t["contracts"] * q
            all_returns.append(t["pnl"] / max(cost, 1e-9))
            all_pnl.append(t["pnl"])
            all_deployed.append(cost)
            all_wins.append(1 if t["won"] else 0)
            bankroll = t["bankroll"]
            equity.append(bankroll)
        if not np.isnan(res.metrics["brier"]):
            all_briers.append(res.metrics["brier"])

    span_years = _span_years(snaps)
    aggregate = compute_metrics(
        trade_returns=np.asarray(all_returns),
        trade_pnl=np.asarray(all_pnl),
        capital_deployed=np.asarray(all_deployed),
        wins=np.asarray(all_wins),
        equity=np.asarray(equity),
        span_years=span_years,
        brier=float(np.mean(all_briers)) if all_briers else float("nan"),
    )

    fold_sharpes = [f["sharpe"] for f in fold_metrics]
    fold_rois = [f["roi"] for f in fold_metrics]
    stability = {
        "n_folds": len(fold_metrics),
        "worst_fold_sharpe": float(min(fold_sharpes)) if fold_sharpes else 0.0,
        "worst_fold_roi": float(min(fold_rois)) if fold_rois else 0.0,
        "frac_profitable_folds":
            float(np.mean([r > 0 for r in fold_rois])) if fold_rois else 0.0,
        "sharpe_std": float(np.std(fold_sharpes)) if len(fold_sharpes) > 1 else 0.0,
    }
    return WalkForwardResult(aggregate=aggregate, folds=fold_metrics,
                             stability=stability, equity_curve=equity)


def _span_years(snaps: list[MarketSnapshot]) -> float:
    if len(snaps) < 2:
        return 1.0
    span = snaps[-1].observed_ts - snaps[0].observed_ts
    return max(span / (365.25 * 24 * 3600.0), 1e-6)
