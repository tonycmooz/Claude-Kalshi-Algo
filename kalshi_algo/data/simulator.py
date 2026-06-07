"""Offline market simulator.

Why this exists
---------------
The live Kalshi API requires authenticated access to historical settlement data
which is not reachable from every environment.  To make the *self-learning loop*
runnable, reproducible and testable offline, we generate a synthetic but
**structurally realistic** universe of binary markets.

Realism is the whole point: a trivial generator would let the model reach 100%
accuracy and the backtest would be meaningless.  Instead we bake in three
well-documented prediction-market inefficiencies and a large dose of irreducible
randomness:

1. **Favorite-longshot bias** - longshots (cheap contracts) are systematically
   *over*-priced and heavy favorites slightly *under*-priced.
2. **Momentum overreaction** - markets that moved sharply recently overshoot
   their fair value before mean-reverting.
3. **Liquidity mispricing** - thin (low-volume, wide-spread) markets are priced
   less efficiently than deep ones.

The market's *quoted price* is therefore a biased, noisy estimate of the latent
true probability ``p_true``; the binary outcome is drawn ``Bernoulli(p_true)``.
A model that sees only the observable features (including the price) can learn
to undo the bias and estimate ``p_true`` better than the raw price - that gap is
the edge the strategy harvests.  Because the outcome is genuinely stochastic,
the predictor's Brier score is floored near ``E[p(1-p)]`` and the strategy can
never be "perfect", exactly as in real life.
"""
from __future__ import annotations

import numpy as np

from .types import MarketSnapshot

# Non-weather categories only, per the mandate.
CATEGORIES = (
    "economics",
    "financials",
    "crypto",
    "politics",
    "companies",
    "sports",
    "entertainment",
)

# Per-category systematic bias (cents) applied to the quoted price.  Represents
# crowd/structural lean that an attentive trader can exploit.
_CATEGORY_BIAS = {
    "economics": -0.010,
    "financials": 0.008,
    "crypto": 0.022,        # crypto markets run hot / over-optimistic
    "politics": 0.014,
    "companies": -0.006,
    "sports": 0.004,
    "entertainment": 0.018,
}

_EPS = 1e-3


class MarketSimulator:
    """Generates a time-ordered stream of resolved markets.

    Parameters
    ----------
    seed:
        RNG seed for reproducibility.
    regime_shift:
        Small multiplier (0 disables) that perturbs the bias coefficients so a
        *live* simulator can differ from the *backtest* one - this is how we
        check the strategy generalizes to an unseen regime rather than overfits.
    """

    def __init__(self, seed: int = 0, regime_shift: float = 0.0):
        self.rng = np.random.default_rng(seed)
        self.regime_shift = regime_shift

    # -- bias coefficients (a regime can nudge them) ------------------------
    def _coef(self, base: float) -> float:
        if self.regime_shift == 0.0:
            return base
        return base * (1.0 + self.regime_shift * self.rng.uniform(-1.0, 1.0))

    def generate(self, n_markets: int = 6000, start_ts: float = 1_700_000_000.0
                 ) -> list[MarketSnapshot]:
        rng = self.rng
        snapshots: list[MarketSnapshot] = []

        # Regime-perturbed structural coefficients (fixed for this universe).
        fl_base = self._coef(0.12)     # favorite-longshot compression strength
        fl_thin = self._coef(0.10)     # extra compression in thin markets
        gamma = self._coef(2.2)        # true outcome sensitivity to the trend
        delta = self._coef(0.8)        # market's (partial) reaction to the trend

        # Markets arrive sequentially in time (one every ~30 min on average).
        ts = start_ts
        for _ in range(n_markets):
            ts += float(rng.exponential(1800.0))
            cat = CATEGORIES[rng.integers(len(CATEGORIES))]

            # Latent true probability: Beta gives a realistic, non-uniform spread
            # of "edges" with mass away from 0.5.
            p_true = float(np.clip(rng.beta(1.5, 1.5), _EPS, 1 - _EPS))

            # Liquidity: heavy-tailed volume; spread shrinks with volume.
            volume = float(np.exp(rng.normal(5.0, 1.1)))         # ~150 median
            open_interest = volume * float(rng.uniform(0.5, 3.0))
            base_spread = float(np.clip(rng.normal(0.025, 0.012), 0.004, 0.12))
            spread = base_spread / (1.0 + np.log1p(volume) / 6.0)

            # Time to close: hours, log-spread from minutes to ~3 weeks.
            hours_to_close = float(np.exp(rng.normal(3.2, 1.3)))
            close_ts = ts + hours_to_close * 3600.0

            # --- structural mispricings (the learnable, exploitable signal) ---
            # The dominant, learnable edge is MOMENTUM UNDERREACTION: a recent
            # trend (an OBSERVABLE signal carried by the price history) genuinely
            # predicts the outcome, but the market price has only partially moved
            # to reflect it.  A model that learns momentum -> outcome therefore
            # estimates the true probability better than the price does.
            mom_sig = float(rng.normal(0.0, 0.06))   # observable trend signal

            # On top of that, a FAVORITE-LONGSHOT compression (the market pulls
            # its estimate toward 0.5) and a per-category lean.
            thinness = 1.0 / (1.0 + np.log1p(volume) / 4.0)      # 0..1, high=thin
            k = float(np.clip(fl_base + fl_thin * thinness, 0.0, 0.4))
            cat_bias = _CATEGORY_BIAS[cat]

            # True probability: fundamentals fully incorporate the trend.
            p_true = float(np.clip(p_true + gamma * mom_sig, _EPS, 1 - _EPS))

            # Market price: fundamentals + only partial trend reaction, then
            # compressed toward 0.5, plus category lean and idiosyncratic noise.
            base = p_true - gamma * mom_sig          # de-trended fundamentals
            price_core = base + delta * mom_sig
            noise = float(rng.normal(0.0, 0.012))
            price = float(np.clip(
                0.5 + (1.0 - k) * (price_core - 0.5) + cat_bias + noise,
                _EPS, 1 - _EPS))

            # Recent price path is consistent with the observable trend signal,
            # so the engineered `momentum` feature recovers `mom_sig`.
            hist = self._price_path(price, mom_sig, rng)

            yes_bid = float(np.clip(price - spread / 2, _EPS, 1 - _EPS))
            yes_ask = float(np.clip(price + spread / 2, _EPS, 1 - _EPS))

            # The actual outcome is genuinely random given p_true.
            result = int(rng.random() < p_true)

            snapshots.append(MarketSnapshot(
                ticker=f"{cat[:3].upper()}-{len(snapshots):06d}",
                category=cat,
                observed_ts=ts,
                close_ts=close_ts,
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                last_price=price,
                volume=volume,
                open_interest=open_interest,
                price_history=hist,
                result=result,
            ))

        return snapshots

    @staticmethod
    def _price_path(price: float, momentum: float, rng: np.random.Generator,
                    n: int = 8) -> tuple[float, ...]:
        """Build a short price history ending at `price` with given drift."""
        path = np.empty(n)
        path[-1] = price
        for i in range(n - 2, -1, -1):
            step = momentum / n + rng.normal(0.0, 0.012)
            path[i] = float(np.clip(path[i + 1] - step, _EPS, 1 - _EPS))
        return tuple(round(float(x), 4) for x in path)
