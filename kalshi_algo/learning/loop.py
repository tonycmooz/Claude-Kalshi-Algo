"""The self-learning loop orchestrator.

Strategy
--------
The loop is an *adaptive* search: it begins with a hand-set baseline, spends an
initial budget **exploring** the space at random, then switches to **exploiting**
- mutating the best configuration found so far with a step size that decays as it
converges (a simulated-annealing flavour).  Every candidate is judged purely on
walk-forward out-of-sample performance, and the loop stops early the moment a
configuration clears all :class:`PerformanceTargets` *and* is robust across folds.

This is the "keep iterating until high performing through validation and testing"
requirement, made concrete and reproducible.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..config import PerformanceTargets, StrategyConfig
from ..data.types import MarketSnapshot
from ..backtest.walkforward import walk_forward_validate, WalkForwardResult
from .optimizer import objective_score, sample_config, mutate_config


@dataclass
class LoopResult:
    best_config: StrategyConfig
    best_wf: WalkForwardResult
    best_score: float
    targets_met: bool
    iterations: int
    history: list[dict] = field(default_factory=list)


class SelfLearningLoop:
    def __init__(
        self,
        validation: list[MarketSnapshot],
        targets: PerformanceTargets | None = None,
        n_folds: int = 4,
        max_iters: int = 40,
        explore_iters: int = 12,
        min_profitable_folds: float = 0.75,
        seed: int = 12,
        time_budget_s: float = 1800.0,
        verbose: bool = True,
    ):
        self.validation = validation
        self.targets = targets or PerformanceTargets()
        self.n_folds = n_folds
        self.max_iters = max_iters
        self.explore_iters = explore_iters
        self.min_profitable_folds = min_profitable_folds
        self.rng = np.random.default_rng(seed)
        self.time_budget_s = time_budget_s
        self.verbose = verbose

    def _evaluate(self, config: StrategyConfig) -> tuple[WalkForwardResult, float]:
        try:
            wf = walk_forward_validate(self.validation, config, n_folds=self.n_folds)
            return wf, objective_score(wf)
        except Exception as exc:  # a bad config shouldn't kill the search
            if self.verbose:
                print(f"   ! candidate failed: {exc}")
            empty = WalkForwardResult(aggregate={"n_trades": 0}, stability={})
            return empty, -1e9

    def _is_high_performing(self, wf: WalkForwardResult) -> bool:
        return (
            self.targets.is_met(wf.aggregate)
            and wf.stability.get("frac_profitable_folds", 0.0) >= self.min_profitable_folds
            and wf.stability.get("worst_fold_roi", -1.0) > 0.0
        )

    def run(self) -> LoopResult:
        start = time.time()
        best_config: StrategyConfig | None = None
        best_wf: WalkForwardResult | None = None
        best_score = -np.inf
        history: list[dict] = []
        targets_met = False
        i = 0

        for i in range(self.max_iters):
            # --- choose the next candidate ---------------------------------
            if i == 0:
                candidate = StrategyConfig()                 # baseline
                phase = "baseline"
            elif best_config is None or i < self.explore_iters or self.rng.random() < 0.2:
                candidate = sample_config(self.rng)          # explore
                phase = "explore"
            else:
                scale = max(0.08, 0.30 * (1.0 - i / self.max_iters))
                candidate = mutate_config(best_config, self.rng, scale)
                phase = "exploit"

            wf, score = self._evaluate(candidate)
            a = wf.aggregate
            improved = score > best_score
            if improved:
                best_score, best_config, best_wf = score, candidate, wf

            history.append({
                "iter": i, "phase": phase, "score": score,
                "sharpe": a.get("sharpe", 0.0), "roi": a.get("roi", 0.0),
                "max_drawdown": a.get("max_drawdown", 0.0),
                "hit_rate": a.get("hit_rate", 0.0), "brier": a.get("brier", float("nan")),
                "n_trades": a.get("n_trades", 0),
                "frac_profitable_folds": wf.stability.get("frac_profitable_folds", 0.0),
                "best_score": best_score,
            })

            if self.verbose:
                print(
                    f"[{i:02d}] {phase:8s} score={score:7.3f} "
                    f"sharpe={a.get('sharpe', 0):6.2f} roi={a.get('roi', 0):6.2%} "
                    f"dd={a.get('max_drawdown', 0):5.1%} "
                    f"hit={a.get('hit_rate', 0):5.1%} "
                    f"brier={a.get('brier', float('nan')):.3f} "
                    f"trades={a.get('n_trades', 0):4d} "
                    f"folds+={wf.stability.get('frac_profitable_folds', 0):.0%}"
                    + ("  <-- best" if improved else "")
                )

            # --- stopping criteria -----------------------------------------
            if self._is_high_performing(wf):
                targets_met = True
                if self.verbose:
                    print(f"\n*** Targets cleared on iteration {i} — high-performing "
                          f"configuration found. ***")
                break
            if time.time() - start > self.time_budget_s:
                if self.verbose:
                    print("\n(time budget exhausted — returning best so far)")
                break

        assert best_config is not None and best_wf is not None
        if self.verbose and not targets_met:
            print(f"\n(reached iteration budget — best score {best_score:.3f})")

        return LoopResult(
            best_config=best_config, best_wf=best_wf, best_score=best_score,
            targets_met=targets_met, iterations=i + 1, history=history,
        )
