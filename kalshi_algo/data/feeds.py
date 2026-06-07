"""Market feeds: a uniform interface over the live exchange and the simulator.

The worker loop depends only on :class:`MarketFeed`, so the exact same loop runs
against Kalshi in production (``KalshiFeed``) and against the simulator offline
(``SimulatedFeed``) for development and tests.  A feed does two things: surface
currently-open markets to trade, and report markets that have newly resolved
(the labels that drive learning).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace as dataclass_replace
from typing import Protocol

from .kalshi_client import KalshiClient
from .types import MarketSnapshot


class MarketFeed(Protocol):
    def fetch_open(self) -> list[MarketSnapshot]: ...
    def fetch_new_resolutions(self) -> list[tuple[str, int, float]]: ...


# --- live exchange -------------------------------------------------------------

class KalshiFeed:
    def __init__(self, client: KalshiClient, max_markets: int = 500):
        self.client = client
        self.max_markets = max_markets

    def fetch_open(self) -> list[MarketSnapshot]:
        return self.client.snapshot_open_markets(max_markets=self.max_markets)

    def fetch_new_resolutions(self) -> list[tuple[str, int, float]]:
        out: list[tuple[str, int, float]] = []
        now = time.time()
        for m in self.client.iter_markets(status="settled",
                                           max_markets=self.max_markets):
            if not self.client.is_non_weather(m):
                continue
            res = str(m.get("result", "")).lower()
            if res not in ("yes", "no"):
                continue
            out.append((m["ticker"], 1 if res == "yes" else 0, now))
        return out


# --- offline replay ------------------------------------------------------------

@dataclass
class SimulatedFeed:
    """Replays a pre-generated market timeline in chronological chunks.

    ``fetch_open`` advances a virtual clock and returns the next batch of markets
    as *unresolved*; ``fetch_new_resolutions`` later reports those whose close
    time has passed - reproducing the real-world lag between observing a market
    and learning its outcome.
    """
    snapshots: list[MarketSnapshot]
    markets_per_cycle: int = 200

    _pos: int = 0
    _now: float = 0.0
    _pending: dict[str, MarketSnapshot] = field(default_factory=dict)
    _resolved: set[str] = field(default_factory=set)
    _added_last: set[str] = field(default_factory=set)

    def fetch_open(self) -> list[MarketSnapshot]:
        batch = self.snapshots[self._pos:self._pos + self.markets_per_cycle]
        self._pos += len(batch)
        if batch:
            self._now = max(s.observed_ts for s in batch)
        out = []
        added: set[str] = set()
        for s in batch:
            self._pending[s.ticker] = s        # remember the true result
            added.add(s.ticker)
            out.append(dataclass_replace(s, result=None))
        # Markets first seen this round cannot also resolve this round, so the
        # loop never opens a position on an already-settled market.
        self._added_last = added
        return out

    def fetch_new_resolutions(self) -> list[tuple[str, int, float]]:
        out: list[tuple[str, int, float]] = []
        for ticker, s in list(self._pending.items()):
            if ticker in self._resolved or ticker in self._added_last:
                continue
            if s.close_ts <= self._now and s.result is not None:
                out.append((ticker, int(s.result), s.close_ts))
                self._resolved.add(ticker)
        return out

    @property
    def exhausted(self) -> bool:
        return self._pos >= len(self.snapshots) and not (
            set(self._pending) - self._resolved)
