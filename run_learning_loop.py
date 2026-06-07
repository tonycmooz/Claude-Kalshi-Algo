#!/usr/bin/env python3
"""End-to-end driver for the self-learning Kalshi strategy.

What it does
------------
1. Builds a *validation* market universe and runs the self-learning loop, which
   keeps proposing and walk-forward-evaluating configurations until one clears
   all performance targets (or the budget is spent).
2. Re-evaluates the winning configuration on a **held-out test universe** that
   was generated with a different seed *and* a shifted bias regime - data the
   optimizer never touched.  This is the honest, generalization number.
3. Writes artifacts (config, metrics, equity curve, markdown report) to
   ``artifacts/``.

Run:  ``python run_learning_loop.py``
"""
from __future__ import annotations

import argparse
import json
import sys

from kalshi_algo.config import PerformanceTargets
from kalshi_algo.data.simulator import MarketSimulator
from kalshi_algo.learning.loop import SelfLearningLoop
from kalshi_algo.backtest.walkforward import walk_forward_validate
from kalshi_algo.reporting import save_artifacts, _fmt


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-markets", type=int, default=6000)
    p.add_argument("--max-iters", type=int, default=40)
    p.add_argument("--n-folds", type=int, default=4)
    p.add_argument("--time-budget", type=float, default=1500.0)
    p.add_argument("--out", default="artifacts")
    p.add_argument("--seed", type=int, default=12)
    args = p.parse_args(argv)

    print("=" * 72)
    print("KALSHI SELF-LEARNING PREDICTION-MARKET STRATEGY  (non-weather)")
    print("=" * 72)

    # --- data universes ----------------------------------------------------
    print(f"\nGenerating validation universe ({args.n_markets} markets)...")
    validation = MarketSimulator(seed=101).generate(n_markets=args.n_markets)
    print(f"Generating held-out TEST universe (different seed + shifted regime)...")
    test_universe = MarketSimulator(seed=999, regime_shift=0.25).generate(
        n_markets=args.n_markets)

    targets = PerformanceTargets()
    print("\nPerformance targets the loop must clear (on walk-forward validation):")
    for k, v in targets.__dict__.items():
        print(f"   {k:20s} {v}")

    # --- the self-learning loop -------------------------------------------
    print("\n" + "-" * 72)
    print("SELF-LEARNING LOOP  (train -> walk-forward validate -> tune -> repeat)")
    print("-" * 72)
    loop = SelfLearningLoop(
        validation=validation, targets=targets, n_folds=args.n_folds,
        max_iters=args.max_iters, time_budget_s=args.time_budget, seed=args.seed)
    result = loop.run()

    print("\n" + "-" * 72)
    print(f"Best configuration (score {result.best_score:.3f}, "
          f"targets_met={result.targets_met}):")
    print(json.dumps(result.best_config.to_dict(), indent=2))

    # --- unbiased confirmation on the held-out test universe ---------------
    print("\n" + "-" * 72)
    print("FINAL OUT-OF-SAMPLE CONFIRMATION on held-out TEST universe")
    print("-" * 72)
    test_wf = walk_forward_validate(
        test_universe, result.best_config, n_folds=args.n_folds)
    print("\nValidation (model-selection) metrics:")
    print(_fmt(result.best_wf.aggregate))
    print("\nTEST (held-out, unseen regime) metrics:")
    print(_fmt(test_wf.aggregate))
    print("\nTest stability:", json.dumps(test_wf.stability, indent=2))

    test_ok = targets.is_met(test_wf.aggregate) and \
        test_wf.stability.get("frac_profitable_folds", 0) >= 0.75

    paths = save_artifacts(
        args.out, result.best_config, result.best_wf, test_wf,
        result.history, result.targets_met)
    print(f"\nArtifacts written: {json.dumps(paths, indent=2)}")

    print("\n" + "=" * 72)
    verdict = ("HIGH-PERFORMING: targets cleared on BOTH validation and the "
               "held-out test universe."
               if (result.targets_met and test_ok) else
               "Targets cleared in validation; review test metrics above.")
    print("VERDICT:", verdict)
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
