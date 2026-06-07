"""Live/paper trading loop with online (continual) learning.

The same :class:`Strategy` that won the backtest is deployed unchanged - this is
the point of keeping the data source abstract.  Each cycle the trader:

1. pulls open, non-weather markets from Kalshi,
2. asks the strategy for sized signals,
3. submits them as **paper** orders (or live, if explicitly enabled), and
4. as markets resolve, books the realized PnL and appends the now-labelled
   snapshot to a training buffer.

When enough fresh outcomes have accrued, :meth:`maybe_retrain` refits the model
so the system keeps *learning from live trading*, not just the initial backtest.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..backtest.engine import kalshi_fee
from ..config import StrategyConfig
from ..data.kalshi_client import KalshiClient
from ..data.types import MarketSnapshot
from ..pipeline import train_strategy
from ..strategy import Strategy

log = logging.getLogger(__name__)


@dataclass
class OpenPosition:
    ticker: str
    side: str
    contracts: int
    entry_price: float
    fee: float


@dataclass
class PaperTrader:
    client: KalshiClient
    strategy: Strategy
    config: StrategyConfig
    bankroll: float = 10_000.0
    retrain_every: int = 250

    open_positions: dict[str, OpenPosition] = field(default_factory=dict)
    realized_pnl: float = 0.0
    training_buffer: list[MarketSnapshot] = field(default_factory=list)
    _since_retrain: int = 0

    # -- one trading cycle --------------------------------------------------
    def step(self, max_markets: int = 300) -> list[OpenPosition]:
        snapshots = self.client.snapshot_open_markets(max_markets=max_markets)
        snapshots = [s for s in snapshots if s.ticker not in self.open_positions]
        if not snapshots:
            return []

        opened: list[OpenPosition] = []
        for sig in self.strategy.generate(snapshots):
            risk_dollars = sig.kelly_fraction * self.bankroll
            contracts = int(risk_dollars // sig.entry_price)
            if contracts < 1:
                continue
            fee = kalshi_fee(contracts, sig.entry_price)
            cost = contracts * sig.entry_price + fee
            if cost > self.bankroll:
                continue

            price_cents = int(round(sig.entry_price * 100))
            self.client.create_order(
                ticker=sig.ticker, side=sig.side, action="buy",
                count=contracts, price_cents=price_cents)
            self.bankroll -= cost
            pos = OpenPosition(sig.ticker, sig.side, contracts,
                               sig.entry_price, fee)
            self.open_positions[sig.ticker] = pos
            opened.append(pos)
            log.info("opened %s x%d @ %.2f (%s)", sig.ticker, contracts,
                     sig.entry_price, sig.side)
        return opened

    # -- settle resolved markets -------------------------------------------
    def reconcile(self, resolved: list[MarketSnapshot]) -> float:
        """Book PnL for resolved markets and feed them to the learner."""
        booked = 0.0
        for s in resolved:
            self.training_buffer.append(s)
            self._since_retrain += 1
            pos = self.open_positions.pop(s.ticker, None)
            if pos is None or s.result is None:
                continue
            won = (pos.side == "yes" and s.result == 1) or \
                  (pos.side == "no" and s.result == 0)
            payout = pos.contracts * 1.0 if won else 0.0
            self.bankroll += payout
            pnl = payout - (pos.contracts * pos.entry_price + pos.fee)
            self.realized_pnl += pnl
            booked += pnl
        return booked

    # -- continual learning -------------------------------------------------
    def maybe_retrain(self) -> bool:
        if self._since_retrain < self.retrain_every:
            return False
        labelled = [s for s in self.training_buffer if s.result is not None]
        if len(labelled) < self.retrain_every:
            return False
        log.info("retraining on %d fresh outcomes", len(labelled))
        self.strategy = train_strategy(labelled, self.config)
        self._since_retrain = 0
        return True
