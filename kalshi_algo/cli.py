"""Command-line interface.

Subcommands
-----------
* ``learn``    - run the self-learning loop and write artifacts.
* ``backtest`` - walk-forward evaluate a single (default or saved) config.
* ``live``     - run the paper trader against Kalshi (needs network/credentials).

Examples
--------
    python -m kalshi_algo.cli learn --n-markets 6000
    python -m kalshi_algo.cli backtest --config artifacts/best_config.json
    python -m kalshi_algo.cli live --cycles 1            # paper mode (default)
    python -m kalshi_algo.cli live --live --cycles 1     # real orders (danger)
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import StrategyConfig, ModelConfig, PerformanceTargets
from .data.simulator import MarketSimulator
from .backtest.walkforward import walk_forward_validate
from .reporting import _fmt


def _load_config(path: str | None) -> StrategyConfig:
    if not path:
        return StrategyConfig()
    with open(path) as f:
        d = json.load(f)
    model = ModelConfig(**d.pop("model"))
    d.pop("feature_set", None)
    return StrategyConfig(model=model, **d)


def cmd_learn(args: argparse.Namespace) -> int:
    from run_learning_loop import main as run_main
    return run_main([
        "--n-markets", str(args.n_markets), "--max-iters", str(args.max_iters),
        "--n-folds", str(args.n_folds), "--out", args.out,
    ])


def cmd_backtest(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    snaps = MarketSimulator(seed=args.seed).generate(args.n_markets)
    wf = walk_forward_validate(snaps, cfg, n_folds=args.n_folds)
    print("Walk-forward aggregate metrics:")
    print(_fmt(wf.aggregate))
    print("\nStability:", json.dumps(wf.stability, indent=2))
    print("\nTargets met:", PerformanceTargets().is_met(wf.aggregate))
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    from .data.kalshi_client import KalshiClient
    from .live.paper_trader import PaperTrader
    from .pipeline import train_strategy

    cfg = _load_config(args.config)
    # Train on simulated history (or plug in real resolved markets here).
    history = MarketSimulator(seed=7).generate(4000)
    strategy = train_strategy(history, cfg)

    client = KalshiClient(paper=not args.live)
    trader = PaperTrader(client=client, strategy=strategy, config=cfg)
    print(f"Paper trader ready (paper={not args.live}).")
    try:
        client.exchange_status()
    except Exception as exc:  # network not available in many environments
        print(f"NOTE: could not reach Kalshi ({exc}).")
        print("The trader is wired correctly; run it where the API is reachable "
              "with valid credentials to trade.")
        return 0
    for c in range(args.cycles):
        opened = trader.step()
        print(f"cycle {c}: opened {len(opened)} positions, "
              f"bankroll=${trader.bankroll:,.2f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="kalshi_algo", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("learn", help="run the self-learning loop")
    pl.add_argument("--n-markets", type=int, default=6000)
    pl.add_argument("--max-iters", type=int, default=40)
    pl.add_argument("--n-folds", type=int, default=4)
    pl.add_argument("--out", default="artifacts")
    pl.set_defaults(func=cmd_learn)

    pb = sub.add_parser("backtest", help="walk-forward evaluate one config")
    pb.add_argument("--config", default=None)
    pb.add_argument("--n-markets", type=int, default=6000)
    pb.add_argument("--n-folds", type=int, default=4)
    pb.add_argument("--seed", type=int, default=101)
    pb.set_defaults(func=cmd_backtest)

    pv = sub.add_parser("live", help="run the (paper) trader against Kalshi")
    pv.add_argument("--config", default=None)
    pv.add_argument("--live", action="store_true", help="ACTUALLY trade (danger)")
    pv.add_argument("--cycles", type=int, default=1)
    pv.set_defaults(func=cmd_live)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
