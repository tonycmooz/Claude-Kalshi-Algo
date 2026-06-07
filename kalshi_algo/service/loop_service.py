"""The orchestrated trading + continual-learning cycle.

One ``run_cycle()`` performs the full live loop:

1. **Collect**  - pull open markets, enrich with news sentiment, persist.
2. **Reconcile** - record new resolutions, settle matching open trades, book PnL.
3. **Learn**    - every N cycles, retrain a *challenger* on all accumulated
   real outcomes, walk-forward validate it, and **promote** it to champion only
   if it beats the incumbent out-of-sample (champion/challenger).
4. **Trade**    - the champion generates signals on open markets; orders are
   sized, **risk-capped**, and submitted (paper by default, live iff enabled).

State (bankroll, daily PnL, cycle count) and all records live in the database so
the loop survives Railway redeploys.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from ..backtest.engine import kalshi_fee
from ..backtest.walkforward import walk_forward_validate
from ..config import PerformanceTargets, StrategyConfig
from ..data.feeds import MarketFeed
from ..data.news import NewsProvider, NullNewsProvider
from ..learning.loop import SelfLearningLoop
from ..pipeline import train_strategy
from ..runtime_config import RuntimeConfig
from ..store import MarketStore, ModelRegistry
from ..strategy import Strategy

log = logging.getLogger(__name__)


@dataclass
class TradingLoop:
    config: RuntimeConfig
    store: MarketStore
    registry: ModelRegistry
    feed: MarketFeed
    news: NewsProvider = field(default_factory=NullNewsProvider)
    targets: PerformanceTargets = field(default_factory=PerformanceTargets)

    strategy: Optional[Strategy] = None
    bankroll: float = 0.0
    cycle: int = 0

    # -- lifecycle ----------------------------------------------------------
    def bootstrap(self) -> None:
        self.bankroll = float(self.store.get_state(
            "bankroll", self.config.starting_bankroll))
        self.cycle = int(self.store.get_state("cycle", 0))
        self.strategy = self.registry.load_active()
        log.info("bootstrap: bankroll=%.2f cycle=%d active_model=%s",
                 self.bankroll, self.cycle, self.strategy is not None)

    # -- one full cycle -----------------------------------------------------
    def run_cycle(self) -> dict:
        self.cycle += 1
        opened = settled = 0
        open_snaps = self._collect()
        settled = self._reconcile()

        if (self.cycle % self.config.learn_every_cycles == 0
                or self.strategy is None):
            self._learn_and_maybe_promote()

        if self.strategy is not None:
            opened = self._trade(open_snaps)

        self.store.set_state("bankroll", self.bankroll)
        self.store.set_state("cycle", self.cycle)
        self.store.set_state("last_cycle_ts", time.time())
        n_open = len(self.store.open_trades())
        self.store.record_equity(self.bankroll, self._realized_pnl_total(),
                                 n_open, self.config.live_trading)
        status = {
            "cycle": self.cycle, "bankroll": round(self.bankroll, 2),
            "opened": opened, "settled": settled, "open_positions": n_open,
            "labels": self.store.count_labels(),
            "has_model": self.strategy is not None,
            "live": self.config.live_trading,
        }
        log.info("cycle %s", status)
        return status

    # -- steps --------------------------------------------------------------
    def _collect(self) -> list:
        snaps = self.feed.fetch_open()
        if not snaps:
            return []
        try:
            self.news.enrich(snaps)
        except Exception as exc:  # news is best-effort
            log.warning("news enrich failed: %s", exc)
        self.store.save_snapshots(snaps)
        return snaps

    def _reconcile(self) -> int:
        resolutions = self.feed.fetch_new_resolutions()
        if not resolutions:
            return 0
        result_by_ticker = {}
        for ticker, result, ts in resolutions:
            self.store.save_resolution(ticker, result, ts)
            result_by_ticker[ticker] = result
        settled = 0
        for t in self.store.open_trades():
            res = result_by_ticker.get(t["ticker"])
            if res is None:
                continue
            won = (t["side"] == "yes" and res == 1) or \
                  (t["side"] == "no" and res == 0)
            payout = t["contracts"] * 1.0 if won else 0.0
            pnl = payout - (t["contracts"] * t["entry_price"] + (t["fee"] or 0.0))
            self.bankroll += payout    # cost was already deducted at entry
            self.store.settle_trade(t["id"], res, pnl)
            self._book_daily_pnl(pnl)
            settled += 1
        return settled

    def _learn_and_maybe_promote(self) -> None:
        data = self.store.fetch_training_examples()
        if len(data) < self.config.min_labels_to_train:
            log.info("learn: only %d labels (<%d) - skipping",
                     len(data), self.config.min_labels_to_train)
            return

        # A bounded search so a worker cycle stays cheap; grows more thorough as
        # data accumulates.  Judged on walk-forward out-of-sample score.
        n_folds = 3 if len(data) < 4000 else 4
        loop = SelfLearningLoop(
            validation=data, targets=self.targets, n_folds=n_folds,
            max_iters=8, explore_iters=4, time_budget_s=240.0,
            seed=self.cycle, verbose=False)
        result = loop.run()
        challenger_score = result.best_score
        active_score = self.registry.active_score()
        agg = result.best_wf.aggregate

        gates_ok = (agg.get("n_trades", 0) >= 50 and agg.get("roi", -1) > 0
                    and agg.get("sharpe", -1) > 0.5 and agg.get("brier", 1) < 0.25)
        promote = gates_ok and (active_score is None
                                or challenger_score > active_score + 0.05)

        strategy = train_strategy(data, result.best_config)
        model_id = self.registry.save(
            strategy, score=challenger_score, validation=agg,
            test=result.best_wf.stability, n_train=len(data), activate=promote)
        log.info("learn: challenger %s score=%.3f active=%.3s promote=%s",
                 model_id, challenger_score, str(active_score), promote)
        if promote:
            self.strategy = strategy

    def _trade(self, open_snaps: list) -> int:
        if not self.config.trading_enabled or not open_snaps:
            return 0
        # Daily loss kill-switch (applies to live; paper keeps learning).
        live = self.config.live_trading and not self._daily_loss_breached()
        caps = self.config.risk
        active_model = self.store.get_active_model()
        model_id = active_model["id"] if active_model else None

        # Avoid duplicate exposure to a market we already hold.
        candidates = [s for s in open_snaps if not self.store.has_open_trade(s.ticker)]
        signals = self.strategy.generate(candidates)

        n_open = len(self.store.open_trades())
        exposure = self._open_exposure()
        opened = 0
        for sig in signals:
            if n_open >= caps.max_open_positions:
                break
            risk_dollars = min(sig.kelly_fraction * self.bankroll,
                               caps.max_position_dollars)
            contracts = int(risk_dollars // sig.entry_price)
            contracts = min(contracts, caps.max_contracts_per_order)
            if contracts < 1:
                continue
            fee = kalshi_fee(contracts, sig.entry_price)
            cost = contracts * sig.entry_price + fee
            if cost > self.bankroll or exposure + cost > caps.max_total_exposure:
                continue

            self._submit(sig, contracts, live)
            self.bankroll -= cost
            self.store.record_trade(
                ticker=sig.ticker, side=sig.side, contracts=contracts,
                entry_price=sig.entry_price, fee=fee, live=live, model_id=model_id)
            exposure += cost
            n_open += 1
            opened += 1
        return opened

    def _submit(self, sig, contracts: int, live: bool) -> None:
        """Send the order to the exchange when live; paper is bookkeeping-only."""
        client = getattr(self.feed, "client", None)
        if live and client is not None:
            client.create_order(
                ticker=sig.ticker, side=sig.side, action="buy",
                count=contracts, price_cents=int(round(sig.entry_price * 100)))

    # -- daily PnL bookkeeping ---------------------------------------------
    def _today(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d")

    def _book_daily_pnl(self, pnl: float) -> None:
        d = self.store.get_state("daily", {"date": self._today(), "pnl": 0.0})
        if d.get("date") != self._today():
            d = {"date": self._today(), "pnl": 0.0}
        d["pnl"] = d.get("pnl", 0.0) + pnl
        self.store.set_state("daily", d)

    def _daily_loss_breached(self) -> bool:
        d = self.store.get_state("daily", {"date": self._today(), "pnl": 0.0})
        return (d.get("date") == self._today()
                and d.get("pnl", 0.0) <= -self.config.risk.daily_loss_limit)

    # -- helpers ------------------------------------------------------------
    def _open_exposure(self) -> float:
        return sum(t["contracts"] * t["entry_price"] + (t["fee"] or 0.0)
                   for t in self.store.open_trades())

    def _realized_pnl_total(self) -> float:
        return self.bankroll - self.config.starting_bankroll + self._open_exposure()
