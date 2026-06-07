"""Unit + integration tests for the strategy stack.

These cover the correctness-critical pieces: no-leakage feature building, fee and
PnL accounting, the favorite-longshot/momentum edge being real (model beats price
out-of-sample), Kelly sizing, drawdown maths, and the end-to-end walk-forward.
"""
from __future__ import annotations

import numpy as np
import pytest

from kalshi_algo.config import StrategyConfig
from kalshi_algo.data.simulator import MarketSimulator, CATEGORIES
from kalshi_algo.data.types import MarketSnapshot
from kalshi_algo.features import FeatureBuilder, NUMERIC_FEATURES
from kalshi_algo.pipeline import train_strategy
from kalshi_algo.backtest.engine import Backtester, kalshi_fee
from kalshi_algo.backtest.metrics import max_drawdown
from kalshi_algo.backtest.walkforward import walk_forward_validate
from kalshi_algo.strategy.signal import _kelly_fraction


# --- data ----------------------------------------------------------------------

def test_simulator_excludes_weather():
    snaps = MarketSimulator(seed=0).generate(500)
    cats = {s.category for s in snaps}
    assert cats.issubset(set(CATEGORIES))
    assert "weather" not in cats and "climate" not in cats


def test_simulator_is_time_ordered_and_resolved():
    snaps = MarketSimulator(seed=0).generate(300)
    ts = [s.observed_ts for s in snaps]
    assert ts == sorted(ts)
    assert all(s.result in (0, 1) for s in snaps)
    assert all(0.0 < s.mid < 1.0 for s in snaps)


# --- features ------------------------------------------------------------------

def test_features_no_leakage_and_aligned():
    snaps = MarketSimulator(seed=1).generate(200)
    fb = FeatureBuilder().fit(snaps)
    X = fb.transform(snaps)
    # The label must never appear as a column.
    assert "result" not in X.columns
    for f in NUMERIC_FEATURES:
        assert f in X.columns
    # Transform on an unseen-category snapshot keeps the schema stable.
    odd = MarketSnapshot(ticker="X", category="newthing", observed_ts=1, close_ts=2,
                         yes_bid=0.4, yes_ask=0.45, last_price=0.42, volume=100,
                         open_interest=100, price_history=(0.4, 0.42), result=1)
    X2 = fb.transform([odd])
    assert list(X2.columns) == list(X.columns)


# --- exchange mechanics --------------------------------------------------------

def test_kalshi_fee_is_rounded_up_and_zero_at_extremes():
    assert kalshi_fee(100, 0.5) == pytest.approx(np.ceil(0.07 * 100 * 0.25 * 100) / 100)
    assert kalshi_fee(10, 0.0) == 0.0
    assert kalshi_fee(10, 1.0) == 0.0


def test_kelly_fraction_math():
    # Edge of zero -> no stake; positive edge -> positive stake bounded by 1.
    assert _kelly_fraction(0.5, 0.5) == 0.0
    assert _kelly_fraction(0.3, 0.5) == 0.0           # negative edge clipped to 0
    f = _kelly_fraction(0.6, 0.5)
    assert 0.0 < f < 1.0


def test_max_drawdown():
    eq = np.array([100, 120, 90, 110, 60, 80], dtype=float)
    # Peak 120 -> trough 60 = 50% drawdown.
    assert max_drawdown(eq) == pytest.approx(0.5)
    assert max_drawdown(np.array([1.0, 2.0, 3.0])) == 0.0


# --- the edge is real ----------------------------------------------------------

def test_model_beats_price_on_traded_edges():
    """Where the model signals a big edge, realized outcomes must move its way.

    This is the core scientific claim: the strategy has genuine predictive edge,
    not just curve-fit noise.
    """
    snaps = MarketSimulator(seed=2).generate(6000)
    n = int(len(snaps) * 0.7)
    strat = train_strategy(snaps[:n], StrategyConfig())
    test = snaps[n:]
    y = np.array([s.result for s in test], float)
    price = np.array([s.mid for s in test])
    p = strat.predictor.predict_proba(strat.fb.transform(test).values)
    edge = p - price
    up = edge > 0.05
    down = edge < -0.05
    assert up.sum() > 30 and down.sum() > 30
    # When the model says YES is cheap, realized YES-rate exceeds the price.
    assert y[up].mean() > price[up].mean()
    # When the model says NO, realized YES-rate is below the price.
    assert y[down].mean() < price[down].mean()


# --- end to end ----------------------------------------------------------------

def test_backtest_runs_and_is_causal():
    snaps = MarketSimulator(seed=3).generate(2000)
    n = int(len(snaps) * 0.7)
    strat = train_strategy(snaps[:n], StrategyConfig())
    res = Backtester(StrategyConfig()).run(strat, snaps[n:])
    m = res.metrics
    assert m["n_trades"] > 0
    assert len(res.equity_curve) == m["n_trades"] + 1
    assert 0.0 <= m["hit_rate"] <= 1.0
    assert m["max_drawdown"] >= 0.0


def test_walk_forward_is_profitable_with_good_config():
    snaps = MarketSimulator(seed=101).generate(6000)
    cfg = StrategyConfig(edge_threshold=0.09, kelly_fraction=0.06,
                         max_position_frac=0.01)
    wf = walk_forward_validate(snaps, cfg, n_folds=4)
    a = wf.aggregate
    assert a["n_trades"] > 200
    assert a["roi"] > 0.0
    assert a["sharpe"] > 1.0
    assert a["brier"] < 0.25
    assert wf.stability["frac_profitable_folds"] >= 0.75


# --- paper trading -------------------------------------------------------------

def test_paper_trader_books_pnl_and_buffers_for_learning():
    from kalshi_algo.data.kalshi_client import KalshiClient
    from kalshi_algo.live.paper_trader import PaperTrader, OpenPosition

    snaps = MarketSimulator(seed=5).generate(2000)
    strat = train_strategy(snaps, StrategyConfig())
    trader = PaperTrader(client=KalshiClient(paper=True), strategy=strat,
                         config=StrategyConfig(), bankroll=1000.0)

    # Manually open a YES position, then resolve it as a win.
    trader.open_positions["T1"] = OpenPosition("T1", "yes", 10, 0.40, 0.07)
    win = MarketSnapshot(ticker="T1", category="crypto", observed_ts=1, close_ts=2,
                         yes_bid=0.4, yes_ask=0.42, last_price=0.41, volume=100,
                         open_interest=100, result=1)
    booked = trader.reconcile([win])
    # Win pays 10 * $1 minus the $4 + $0.07 cost basis = $5.93.
    assert booked == pytest.approx(10 * 1.0 - (10 * 0.40 + 0.07))
    assert "T1" not in trader.open_positions
    assert len(trader.training_buffer) == 1  # available for continual learning
