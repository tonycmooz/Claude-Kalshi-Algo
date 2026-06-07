# Kalshi Strategy - Self-Learning Run Report

**Acceptance status:** PASSED
**Search iterations:** 3

## Validation (walk-forward, used for model selection)
  n_trades        : 1507.000
  roi             :    8.15%
  total_return    :  176.54%
  sharpe          :    4.594
  sortino         :  425.788
  hit_rate        :   58.59%
  profit_factor   :    1.196
  max_drawdown    :   16.22%
  brier           :    0.190
  final_equity    : $27,654

Stability: {
  "n_folds": 4,
  "worst_fold_sharpe": 2.7332033979103136,
  "worst_fold_roi": 0.024993086908374644,
  "frac_profitable_folds": 1.0,
  "sharpe_std": 4.209997489287781
}

## Test (held-out universe, unseen regime - the unbiased number)
  n_trades        : 1782.000
  roi             :   12.74%
  total_return    :  743.25%
  sharpe          :   10.481
  sortino         :  854.410
  hit_rate        :   69.53%
  profit_factor   :    1.424
  max_drawdown    :   12.82%
  brier           :    0.182
  final_equity    : $84,325

Stability: {
  "n_folds": 4,
  "worst_fold_sharpe": 9.28857405441681,
  "worst_fold_roi": 0.11430151663528339,
  "frac_profitable_folds": 1.0,
  "sharpe_std": 2.072045707240463
}

## Selected configuration
```json
{
  "model": {
    "n_estimators": 300,
    "learning_rate": 0.02422012878922756,
    "max_depth": 4,
    "min_samples_leaf": 40,
    "subsample": 0.766758416253241,
    "max_features": 0.7268080609266383,
    "calibration": "sigmoid",
    "random_state": 7
  },
  "edge_threshold": 0.09492616906106284,
  "kelly_fraction": 0.08657879609598708,
  "max_position_frac": 0.009697255269613448,
  "min_price": 0.07693573318381501,
  "max_price": 0.9662633092346384,
  "min_volume": 50.0,
  "max_spread": 0.07614054377218757,
  "allow_short_no": true,
  "feature_set": []
}
```

> The *test* block is the headline result: it is computed on a market universe
> generated with a different seed *and* a shifted bias regime that the optimizer
> never saw, so it measures genuine generalization rather than in-sample fit.
