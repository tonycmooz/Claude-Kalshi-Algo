"""Artifact + report writers for a finished optimization run."""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any

from .config import StrategyConfig
from .backtest.walkforward import WalkForwardResult


def _fmt(m: dict[str, float]) -> str:
    order = ["n_trades", "roi", "total_return", "sharpe", "sortino", "hit_rate",
             "profit_factor", "max_drawdown", "brier", "final_equity"]
    lines = []
    for k in order:
        if k not in m:
            continue
        v = m[k]
        if k in ("roi", "total_return", "hit_rate", "max_drawdown"):
            lines.append(f"  {k:16s}: {v:8.2%}")
        elif k in ("final_equity",):
            lines.append(f"  {k:16s}: ${v:,.0f}")
        else:
            lines.append(f"  {k:16s}: {v:8.3f}")
    return "\n".join(lines)


def save_artifacts(out_dir: str, config: StrategyConfig,
                   validation: WalkForwardResult, test: WalkForwardResult,
                   history: list[dict], targets_met: bool) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths: dict[str, str] = {}

    cfg_path = os.path.join(out_dir, "best_config.json")
    with open(cfg_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    paths["config"] = cfg_path

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "targets_met": targets_met,
            "validation": validation.aggregate,
            "validation_stability": validation.stability,
            "test": test.aggregate,
            "test_stability": test.stability,
        }, f, indent=2)
    paths["metrics"] = metrics_path

    hist_path = os.path.join(out_dir, "search_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    paths["history"] = hist_path

    # Equity curve (out-of-sample test) as CSV.
    eq_path = os.path.join(out_dir, "test_equity_curve.csv")
    with open(eq_path, "w") as f:
        f.write("trade_index,equity\n")
        for i, e in enumerate(test.equity_curve):
            f.write(f"{i},{e:.2f}\n")
    paths["equity_curve"] = eq_path

    report_path = os.path.join(out_dir, "REPORT.md")
    with open(report_path, "w") as f:
        f.write(_report_md(config, validation, test, history, targets_met))
    paths["report"] = report_path
    return paths


def _report_md(config: StrategyConfig, validation: WalkForwardResult,
               test: WalkForwardResult, history: list[dict],
               targets_met: bool) -> str:
    status = "PASSED" if targets_met else "BEST-EFFORT (targets not fully cleared)"
    return f"""# Kalshi Strategy - Self-Learning Run Report

**Acceptance status:** {status}
**Search iterations:** {len(history)}

## Validation (walk-forward, used for model selection)
{_fmt(validation.aggregate)}

Stability: {json.dumps(validation.stability, indent=2)}

## Test (held-out universe, unseen regime - the unbiased number)
{_fmt(test.aggregate)}

Stability: {json.dumps(test.stability, indent=2)}

## Selected configuration
```json
{json.dumps(config.to_dict(), indent=2)}
```

> The *test* block is the headline result: it is computed on a market universe
> generated with a different seed *and* a shifted bias regime that the optimizer
> never saw, so it measures genuine generalization rather than in-sample fit.
"""
