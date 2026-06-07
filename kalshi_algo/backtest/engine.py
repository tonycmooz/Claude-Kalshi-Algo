"""Event-driven backtester.

The exchange is modelled honestly:

* trades are taken at the **ask/bid** (the spread is paid, via the signal),
* Kalshi's per-contract trading **fee** is deducted on entry,
* contracts settle at $1 (win) or $0 (loss) at resolution,
* capital **compounds**: each bet is sized off the *current* bankroll, so a bad
  run shrinks subsequent bets (the Kelly safety valve) and the equity curve is
  realistic.

Markets are processed in chronological order of ``observed_ts`` so there is no
look-ahead, and each market's resolution is only used to settle a position that
was opened from information available strictly before it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..config import KALSHI_FEE_COEFFICIENT, StrategyConfig
from ..data.types import MarketSnapshot
from ..strategy import Strategy
from .metrics import compute_metrics


def kalshi_fee(contracts: int, price: float) -> float:
    """Kalshi general trading fee: round_up(0.07 * C * P * (1-P)) dollars."""
    raw = KALSHI_FEE_COEFFICIENT * contracts * price * (1.0 - price)
    return math.ceil(raw * 100.0) / 100.0


@dataclass
class BacktestResult:
    metrics: dict[str, float]
    equity_curve: list[float] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)


class Backtester:
    def __init__(self, config: StrategyConfig, initial_bankroll: float = 10_000.0):
        self.config = config
        self.initial_bankroll = initial_bankroll

    def run(self, strategy: Strategy, snapshots: list[MarketSnapshot]
            ) -> BacktestResult:
        # Chronological order — strictly causal.
        snaps = sorted(snapshots, key=lambda s: s.observed_ts)
        signals = strategy.generate(snaps)
        signals.sort(key=lambda sig: sig.snapshot.observed_ts)

        bankroll = self.initial_bankroll
        equity = [bankroll]
        returns, pnls, deployed, wins = [], [], [], []
        trades = []

        for sig in signals:
            q = sig.entry_price
            risk_dollars = sig.kelly_fraction * bankroll
            contracts = int(risk_dollars // q)
            if contracts < 1:
                continue
            cost = contracts * q
            fee = kalshi_fee(contracts, q)
            if cost + fee > bankroll:
                continue

            s = sig.snapshot
            won = (sig.side == "yes" and s.result == 1) or \
                  (sig.side == "no" and s.result == 0)
            if won:
                pnl = contracts * (1.0 - q) - fee
            else:
                pnl = -cost - fee

            bankroll += pnl
            equity.append(bankroll)
            returns.append(pnl / (cost + fee))   # return on capital at risk
            pnls.append(pnl)
            deployed.append(cost + fee)
            wins.append(1 if won else 0)
            trades.append({
                "ticker": sig.ticker, "side": sig.side, "price": q,
                "contracts": contracts, "edge": sig.edge,
                "win_prob": sig.win_prob, "pnl": pnl, "won": won,
                "bankroll": bankroll, "ts": s.observed_ts,
            })

            if bankroll <= 0:  # ruin guard
                break

        # Predictor quality (Brier) over *all* evaluated markets, not just traded.
        brier = self._brier(strategy, snaps)
        span_years = self._span_years(snaps)

        metrics = compute_metrics(
            trade_returns=np.asarray(returns),
            trade_pnl=np.asarray(pnls),
            capital_deployed=np.asarray(deployed),
            wins=np.asarray(wins),
            equity=np.asarray(equity),
            span_years=span_years,
            brier=brier,
        )
        return BacktestResult(metrics=metrics, equity_curve=equity, trades=trades)

    @staticmethod
    def _brier(strategy: Strategy, snaps: list[MarketSnapshot]) -> float:
        labelled = [s for s in snaps if s.result is not None]
        if not labelled:
            return float("nan")
        probs = strategy.predictor.predict_proba(
            strategy.fb.transform(labelled).values)
        y = np.array([s.result for s in labelled], dtype=float)
        return float(np.mean((probs - y) ** 2))

    @staticmethod
    def _span_years(snaps: list[MarketSnapshot]) -> float:
        if len(snaps) < 2:
            return 1.0
        span = snaps[-1].observed_ts - snaps[0].observed_ts
        return max(span / (365.25 * 24 * 3600.0), 1e-6)
