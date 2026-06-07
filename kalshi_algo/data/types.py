"""Shared data structures describing a tradeable market observation.

A :class:`MarketSnapshot` is the unit of decision making: it bundles everything
observable about a binary market at the moment we consider trading it, plus the
eventual resolution (``result``) which is only known after the fact and is used
exclusively for training / backtesting (never as a live input feature).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MarketSnapshot:
    # Identity ---------------------------------------------------------------
    ticker: str
    category: str
    # Timing -----------------------------------------------------------------
    observed_ts: float        # unix seconds when the snapshot was taken
    close_ts: float           # unix seconds when the market resolves
    # Microstructure (all prices in dollars, 0..1) ---------------------------
    yes_bid: float
    yes_ask: float
    last_price: float
    volume: float             # contracts traded so far
    open_interest: float
    # Short price history (most-recent-last), used for momentum features -----
    price_history: tuple[float, ...] = ()
    # Outcome (None until the market resolves) -------------------------------
    result: Optional[int] = None   # 1 if YES resolved true, 0 otherwise
    # Optional external signals (e.g. news sentiment) keyed by feature name.
    # Populated by data providers; consumed opportunistically by features.
    extra: dict = field(default_factory=dict)

    @property
    def mid(self) -> float:
        if self.yes_bid > 0 and self.yes_ask > 0:
            return 0.5 * (self.yes_bid + self.yes_ask)
        return self.last_price

    @property
    def spread(self) -> float:
        if self.yes_bid > 0 and self.yes_ask > 0:
            return max(0.0, self.yes_ask - self.yes_bid)
        return 0.0

    @property
    def hours_to_close(self) -> float:
        return max(0.0, (self.close_ts - self.observed_ts) / 3600.0)
