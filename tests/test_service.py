"""Tests for the deployable worker: persistence, feed lag, and the trade loop.

These run entirely offline against SQLite + the simulated feed, mirroring what
the Railway worker does against Postgres + Kalshi.
"""
from __future__ import annotations

import tempfile

import pytest

from kalshi_algo.config import StrategyConfig
from kalshi_algo.data.feeds import SimulatedFeed
from kalshi_algo.data.news import NullNewsProvider
from kalshi_algo.data.simulator import MarketSimulator
from kalshi_algo.pipeline import train_strategy
from kalshi_algo.runtime_config import RiskCaps, RuntimeConfig
from kalshi_algo.service.loop_service import TradingLoop
from kalshi_algo.store import MarketStore, ModelRegistry
from kalshi_algo.store.db import make_engine


def _make_loop(tmp, *, with_model=True, learn_every=10_000):
    cfg = RuntimeConfig(
        data_source="sim", database_url=f"sqlite:///{tmp}/t.db",
        model_dir=f"{tmp}/m", live_trading=False, starting_bankroll=10_000.0,
        cycle_seconds=0, learn_every_cycles=learn_every, min_labels_to_train=500,
        risk=RiskCaps(max_total_exposure=3000, max_position_dollars=60))
    engine = make_engine(cfg.database_url)
    store = MarketStore(engine)
    registry = ModelRegistry(store, cfg.model_dir)
    snaps = MarketSimulator(seed=101).generate(4000)
    feed = SimulatedFeed(snaps, markets_per_cycle=300)
    loop = TradingLoop(config=cfg, store=store, registry=registry, feed=feed,
                       news=NullNewsProvider())
    loop.bootstrap()
    if with_model:
        # Inject a pre-trained champion so the test avoids the slow search.
        strat = train_strategy(MarketSimulator(seed=5).generate(3000), StrategyConfig())
        registry.save(strat, score=1.0, validation={}, test={}, n_train=3000,
                      activate=True)
        loop.strategy = strat
    return loop, store, engine


def test_store_dedupes_snapshots_and_joins_labels():
    with tempfile.TemporaryDirectory() as tmp:
        engine = make_engine(f"sqlite:///{tmp}/t.db")
        store = MarketStore(engine)
        snaps = MarketSimulator(seed=1).generate(300)
        assert store.save_snapshots(snaps) == 300
        assert store.save_snapshots(snaps) == 0          # idempotent
        for s in snaps[:100]:
            store.save_resolution(s.ticker, int(s.result), s.observed_ts)
        assert store.count_labels() == 100
        labelled = store.fetch_training_examples()
        assert len(labelled) == 100
        assert all(s.result in (0, 1) for s in labelled)


def test_simulated_feed_never_resolves_same_cycle_as_open():
    snaps = MarketSimulator(seed=2).generate(2000)
    feed = SimulatedFeed(snaps, markets_per_cycle=400)
    for _ in range(6):
        opened = {s.ticker for s in feed.fetch_open()}
        resolved = {t for t, _, _ in feed.fetch_new_resolutions()}
        # A market revealed this cycle must not also resolve this cycle.
        assert opened.isdisjoint(resolved)


def test_loop_opens_and_settles_trades_and_recycles_capital():
    with tempfile.TemporaryDirectory() as tmp:
        loop, store, engine = _make_loop(tmp)
        for _ in range(12):
            loop.run_cycle()

        import sqlalchemy as sa
        from kalshi_algo.store import db as D
        with engine.begin() as c:
            total = c.execute(sa.select(sa.func.count()).select_from(D.trades)).scalar()
            settled = c.execute(sa.select(sa.func.count()).select_from(D.trades)
                                .where(D.trades.c.status == "settled")).scalar()
        assert total > 20            # actively trading
        assert settled > 10          # positions actually resolve
        # Risk cap respected: never exceed max total exposure.
        assert loop._open_exposure() <= loop.config.risk.max_total_exposure + 1e-6


def test_daily_loss_kill_switch_blocks_live_entries():
    with tempfile.TemporaryDirectory() as tmp:
        loop, store, _ = _make_loop(tmp)
        loop.config = loop.config  # noqa - readability
        # Force a breached daily loss and verify the live gate trips.
        store.set_state("daily", {"date": loop._today(),
                                  "pnl": -loop.config.risk.daily_loss_limit - 1})
        assert loop._daily_loss_breached() is True
