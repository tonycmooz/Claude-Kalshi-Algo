"""Performance metrics for a sequence of resolved trades.

All metrics are computed from realized per-trade PnL and the equity curve, so
they reflect fees and spread, not idealized fair-value moves.
"""
from __future__ import annotations

import numpy as np


def max_drawdown(equity: np.ndarray) -> float:
    """Largest peak-to-trough fractional decline of the equity curve."""
    if equity.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    drawdowns = (running_max - equity) / np.where(running_max > 0, running_max, 1.0)
    return float(np.max(drawdowns))


def compute_metrics(
    trade_returns: np.ndarray,     # per-trade return on bankroll-at-risk basis
    trade_pnl: np.ndarray,         # realized $ PnL per trade
    capital_deployed: np.ndarray,  # $ cost basis per trade
    wins: np.ndarray,              # 1/0 win flags per trade
    equity: np.ndarray,            # equity curve (starts at initial bankroll)
    span_years: float,             # wall-clock span of the trades
    brier: float,                  # predictor Brier score over evaluated markets
) -> dict[str, float]:
    n = int(trade_returns.size)
    if n == 0:
        return {
            "n_trades": 0, "roi": 0.0, "total_return": 0.0, "sharpe": 0.0,
            "sortino": 0.0, "hit_rate": 0.0, "profit_factor": 0.0,
            "max_drawdown": 0.0, "avg_edge_capture": 0.0, "brier": brier,
            "final_equity": float(equity[-1]) if equity.size else 0.0,
        }

    total_pnl = float(np.sum(trade_pnl))
    total_deployed = float(np.sum(capital_deployed))
    roi = total_pnl / total_deployed if total_deployed > 0 else 0.0

    mean_r = float(np.mean(trade_returns))
    std_r = float(np.std(trade_returns, ddof=1)) if n > 1 else 0.0
    downside = trade_returns[trade_returns < 0]
    dstd = float(np.std(downside, ddof=1)) if downside.size > 1 else 0.0

    # Annualize the per-trade Sharpe by the realized trade frequency.
    trades_per_year = n / span_years if span_years > 0 else float(n)
    ann = np.sqrt(max(trades_per_year, 1.0))
    sharpe = (mean_r / std_r * ann) if std_r > 0 else 0.0
    sortino = (mean_r / dstd * ann) if dstd > 0 else 0.0

    gross_win = float(np.sum(trade_pnl[trade_pnl > 0]))
    gross_loss = float(-np.sum(trade_pnl[trade_pnl < 0]))
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

    return {
        "n_trades": n,
        "roi": roi,
        "total_return": float(equity[-1] / equity[0] - 1.0) if equity.size else 0.0,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "hit_rate": float(np.mean(wins)),
        "profit_factor": float(min(profit_factor, 1e6)),
        "max_drawdown": max_drawdown(equity),
        "brier": float(brier),
        "final_equity": float(equity[-1]) if equity.size else 0.0,
    }
