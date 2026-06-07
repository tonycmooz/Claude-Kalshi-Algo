"""From calibrated probabilities to sized, risk-checked trade decisions.

Pipeline per market
-------------------
1. Liquidity / sanity filters (price band, volume, spread) drop untradeable
   markets up front.
2. The predictor gives ``p`` = P(YES).  We compare against the price we would
   actually *pay* to cross the spread:
       - buy YES at ``yes_ask``      -> edge = p        - yes_ask
       - buy NO  at ``1 - yes_bid``  -> edge = (1 - p)  - (1 - yes_bid)
   Using the ask/bid (not the mid) makes the edge net of the half-spread, so the
   backtest cannot earn phantom profits that evaporate on real fills.
3. The better side, if its edge clears ``edge_threshold``, is sized with
   **fractional Kelly** and capped by ``max_position_frac``.

The output carries only *fractions* (of bankroll); the backtester / live trader
converts them to contract counts against the live bankroll.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import StrategyConfig
from ..data.types import MarketSnapshot
from ..features import FeatureBuilder
from ..models import FairValuePredictor


@dataclass
class TradeSignal:
    ticker: str
    side: str                 # "yes" | "no"
    entry_price: float        # dollars paid per contract (the side's ask)
    win_prob: float           # model P(this side wins)
    edge: float               # win_prob - entry_price
    kelly_fraction: float     # fraction of bankroll to put at risk (capped)
    snapshot: MarketSnapshot


def _kelly_fraction(p: float, q: float) -> float:
    """Full-Kelly fraction of bankroll to stake on a binary contract.

    Stake ``q`` to win ``1`` with probability ``p``: f* = (p - q) / (1 - q).
    """
    if q <= 0.0 or q >= 1.0:
        return 0.0
    return max(0.0, (p - q) / (1.0 - q))


class Strategy:
    """Couples a fitted predictor + feature builder with trading rules."""

    def __init__(self, config: StrategyConfig,
                 predictor: FairValuePredictor,
                 feature_builder: FeatureBuilder):
        self.config = config
        self.predictor = predictor
        self.fb = feature_builder

    def _passes_filters(self, s: MarketSnapshot) -> bool:
        c = self.config
        price = s.mid
        return (
            c.min_price <= price <= c.max_price
            and s.volume >= c.min_volume
            and 0.0 <= s.spread <= c.max_spread
            and s.yes_ask > 0 and s.yes_bid > 0
        )

    def generate(self, snapshots: list[MarketSnapshot]) -> list[TradeSignal]:
        if not snapshots:
            return []
        probs = self.predictor.predict_proba(self.fb.transform(snapshots).values)
        c = self.config
        signals: list[TradeSignal] = []

        for s, p in zip(snapshots, probs):
            if not self._passes_filters(s):
                continue

            # Cost to actually take each side, net of the half-spread.
            yes_ask = float(s.yes_ask)
            no_ask = float(1.0 - s.yes_bid)

            yes_edge = p - yes_ask
            no_edge = (1.0 - p) - no_ask if c.allow_short_no else -1.0

            if yes_edge >= no_edge and yes_edge >= c.edge_threshold:
                side, q, win_p, edge = "yes", yes_ask, float(p), float(yes_edge)
            elif c.allow_short_no and no_edge >= c.edge_threshold:
                side, q, win_p, edge = "no", no_ask, float(1.0 - p), float(no_edge)
            else:
                continue

            frac = c.kelly_fraction * _kelly_fraction(win_p, q)
            frac = float(np.clip(frac, 0.0, c.max_position_frac))
            if frac <= 0.0:
                continue

            signals.append(TradeSignal(
                ticker=s.ticker, side=side, entry_price=q, win_prob=win_p,
                edge=edge, kelly_fraction=frac, snapshot=s))

        return signals
