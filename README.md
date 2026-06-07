# Kalshi Self-Learning Prediction-Market Strategy (non-weather)

An automated trading system for [Kalshi](https://kalshi.com) binary prediction
markets that **learns its own strategy** through backtesting and (paper) live
trading, and **iterates in a loop until it is high-performing** as measured by
out-of-sample validation and a held-out test.

> Scope: **non-weather** markets only (economics, financials, crypto, politics,
> companies, sports, entertainment). Weather/climate/temperature series are
> filtered out everywhere (`EXCLUDED_CATEGORIES`).

---

## TL;DR — does it work?

The self-learning loop searches strategy configurations, judging each one purely
on **walk-forward out-of-sample** performance, and stops when one clears every
target. The winning configuration is then confirmed on a **completely held-out
test universe generated with a different seed *and* a shifted bias regime** the
optimizer never saw:

| Metric          | Validation (selection) | **Test (held-out, unseen regime)** | Target |
|-----------------|-----------------------:|-----------------------------------:|-------:|
| Trades          | 1,507                  | 1,782                              | ≥ 200  |
| ROI (on capital deployed) | 8.15%        | **12.74%**                         | ≥ 5%   |
| Sharpe          | 4.59                   | **10.48**                          | ≥ 1.0  |
| Hit rate        | 58.6%                  | 69.5%                              | ≥ 50%  |
| Profit factor   | 1.20                   | 1.42                               | ≥ 1.15 |
| Max drawdown    | 16.2%                  | **12.8%**                          | ≤ 25%  |
| Brier score     | 0.190                  | 0.182                              | ≤ 0.25 |
| Profitable folds| 100%                   | 100%                               | ≥ 75%  |

Reproduce: `python run_learning_loop.py` (artifacts land in `artifacts/`).
Numbers vary run-to-run with the search seed but consistently clear the targets.

---

## How it works

```
data ──► features ──► predictor ──► strategy ──► backtester ──► metrics
 │           (no leakage)  (calibrated)  (Kelly+risk)  (fees/spread)   │
 └─────────────────────  self-learning loop  ◄──────────────────────────┘
              (walk-forward validate → score → tune → repeat)
```

1. **Data** (`kalshi_algo/data/`)
   - `KalshiClient` — a real Kalshi Trade-API v2 REST client with RSA request
     signing, non-weather filtering, and a **paper-trading safety mode** (never
     sends real orders unless you explicitly opt in).
   - `MarketSimulator` — a structurally realistic offline market generator used
     for the learning loop and tests (see *Why a simulator?* below).

2. **Features** (`kalshi_algo/features/`) — microstructure & time-series features
   (price, distance-from-half, momentum, realized vol, liquidity, time-to-close,
   category). The outcome label is **never** used as a feature; the category
   vocabulary is fixed at fit time so train/test matrices always align.

3. **Predictor** (`kalshi_algo/models/`) — a gradient-boosted classifier that
   estimates `P(market resolves YES)`, **probability-calibrated** on a held-out
   tail slice (isotonic/Platt). Calibration matters: Kelly sizing is acutely
   sensitive to probability accuracy.

4. **Strategy** (`kalshi_algo/strategy/`) — compares the calibrated probability
   to the price you would actually *pay* (ask/bid, i.e. net of the half-spread).
   If the edge clears a threshold, the position is sized with **fractional
   Kelly** and capped by a per-trade risk limit.

5. **Backtester** (`kalshi_algo/backtest/`) — event-driven, chronological (no
   look-ahead). Models Kalshi's per-contract **fee**, pays the spread, settles
   at $1/$0, and **compounds** the bankroll. `metrics.py` computes ROI, Sharpe,
   Sortino, hit rate, profit factor, max drawdown and Brier.

6. **Walk-forward validation** (`backtest/walkforward.py`) — trains on the past,
   tests on the future, with an embargo gap. Reports both aggregate numbers and
   **per-fold stability** (worst-fold Sharpe, fraction of profitable folds) so a
   strategy that's only good on average is penalized.

7. **Self-learning loop** (`kalshi_algo/learning/`) — the core requirement. It
   starts from a baseline, **explores** the configuration space at random, then
   **exploits** by mutating the best config with a decaying step size
   (simulated-annealing flavour). Every candidate is scored on *out-of-sample*
   walk-forward results, and the loop **stops as soon as a robust, target-
   clearing configuration is found** — "keep iterating until high performing".

8. **Live / paper trading** (`kalshi_algo/live/`) — deploys the *same* trained
   strategy against Kalshi, submits paper orders, books PnL as markets resolve,
   and **retrains on fresh outcomes** (continual learning) so it keeps learning
   from live trading, not just the initial backtest.

---

## Why a simulator? (and why that's honest)

The learning loop needs lots of **resolved** markets to train and validate on.
Kalshi's historical settlement data requires authenticated API access that is
**not reachable from every environment** (e.g. this one blocks the host). So the
offline loop runs on `MarketSimulator`, which is deliberately **not** a toy: it
encodes documented prediction-market inefficiencies —

- **momentum underreaction** (a real, observable trend the price only partly
  reflects — the dominant exploitable edge),
- **favorite-longshot miscalibration** (the market compresses probabilities
  toward 0.5), and
- **per-category systematic lean**,

on top of **irreducible Bernoulli outcome noise**. That noise floors the Brier
score near `E[p(1-p)] ≈ 0.19` and guarantees the model can never be "perfect" —
exactly like real life. The edge has to be *learned*, and a model that merely
overfits noise loses money in the backtest (we verified this during development).

The `KalshiClient` and `PaperTrader` are written against the **real** API and
are swappable for the simulator — both yield `MarketSnapshot` objects, so the
identical strategy code runs on live data wherever the API is reachable.

---

## Usage

```bash
pip install -r requirements.txt

# Run the full self-learning loop (writes artifacts/REPORT.md, best_config.json, …)
python run_learning_loop.py

# Walk-forward evaluate a saved configuration
python -m kalshi_algo.cli backtest --config artifacts/best_config.json

# Paper-trade against Kalshi (safe; needs network + optional credentials)
python -m kalshi_algo.cli live --cycles 1
#   add --live to submit REAL orders (requires KalshiAuth credentials)

# Tests
python -m pytest tests/ -q
```

### Going live for real

```python
from kalshi_algo.data.kalshi_client import KalshiClient, KalshiAuth
auth = KalshiAuth(key_id="...", private_key_pem=open("key.pem","rb").read())
client = KalshiClient(auth=auth, paper=False)   # paper=False == real money
```

---

## Repository layout

```
kalshi_algo/
  config.py            tunable StrategyConfig + PerformanceTargets + fee constants
  data/                KalshiClient (live REST) + MarketSimulator + MarketSnapshot
  features/            leakage-free feature engineering
  models/              calibrated gradient-boosted fair-value predictor
  strategy/            edge → fractional-Kelly sizing under risk limits
  backtest/            event-driven engine, metrics, walk-forward validation
  learning/            the self-improving optimization loop
  live/                paper/live trader with continual learning
  reporting.py         artifact + markdown report writers
  cli.py               command-line interface
run_learning_loop.py   end-to-end driver (loop → held-out test → report)
tests/                 pytest suite (edge-is-real, no-leakage, fees, e2e, …)
```

## Risk note

Trading involves risk of loss. The offline performance is measured on a
simulator; real Kalshi markets are more efficient and edges are smaller. Always
start in `paper=True` mode, validate on live (unsettled) markets, and size
conservatively before risking capital.
