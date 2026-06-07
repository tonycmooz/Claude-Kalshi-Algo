"""Automated prediction-markets trading system for Kalshi (non-weather markets).

The package is organized into composable layers:

- ``data``     : market data access (live Kalshi REST client + offline simulator)
- ``features`` : feature engineering from market microstructure / time series
- ``models``   : the probabilistic predictor (fair-value estimator) + calibration
- ``strategy`` : turning model probabilities into sized positions under risk limits
- ``backtest`` : event-driven backtester, performance metrics, walk-forward validation
- ``learning`` : the self-improving optimization loop (train -> validate -> tune -> repeat)
- ``live``     : paper/live trading loop that reuses the trained strategy online
"""

__version__ = "0.1.0"
