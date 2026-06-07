"""Railway worker entrypoint.

Wires runtime config -> data source -> persistence -> trading loop, starts the
health server, and then runs the collect/learn/trade cycle forever on a fixed
cadence.  Designed to be the container's main process:

    python -m kalshi_algo.service.worker
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import time

from ..data.feeds import KalshiFeed, SimulatedFeed
from ..data.kalshi_client import KalshiAuth, KalshiClient
from ..data.news import build_news_provider
from ..data.simulator import MarketSimulator
from ..runtime_config import RuntimeConfig
from ..store import MarketStore, ModelRegistry
from ..store.db import make_engine
from .health import start_health_server
from .loop_service import TradingLoop

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("kalshi_algo.worker")


def build_feed(cfg: RuntimeConfig):
    if cfg.data_source == "sim":
        log.info("DATA_SOURCE=sim - replaying a simulated market timeline")
        snaps = MarketSimulator(seed=int(os.getenv("SIM_SEED", "0"))).generate(
            n_markets=int(os.getenv("SIM_MARKETS", "8000")))
        return SimulatedFeed(snaps, markets_per_cycle=int(
            os.getenv("SIM_MARKETS_PER_CYCLE", "200")))

    auth = None
    if cfg.kalshi_key_id and cfg.kalshi_private_key_pem:
        auth = KalshiAuth(cfg.kalshi_key_id, cfg.kalshi_private_key_pem)
        log.info("Kalshi auth configured (key id %s...)", cfg.kalshi_key_id[:6])
    else:
        log.warning("No Kalshi credentials - public data only, paper trading.")
    client = KalshiClient(base_url=cfg.kalshi_base_url, auth=auth,
                          paper=not cfg.live_trading)
    return KalshiFeed(client)


def main() -> int:
    cfg = RuntimeConfig.from_env()
    log.info("starting worker: data_source=%s live=%s db=%s model_dir=%s",
             cfg.data_source, cfg.live_trading,
             cfg.database_url.split("@")[-1], cfg.model_dir)

    engine = make_engine(cfg.database_url)
    store = MarketStore(engine)
    registry = ModelRegistry(store, cfg.model_dir)
    feed = build_feed(cfg)
    news = build_news_provider(cfg.news_api_url, cfg.news_api_key)

    loop = TradingLoop(config=cfg, store=store, registry=registry,
                       feed=feed, news=news)
    loop.bootstrap()

    last_status = {"status": "starting", "cycle": loop.cycle}
    start_health_server(cfg.port, lambda: last_status)

    stop = {"flag": False}

    def _graceful(signum, _frame):
        log.info("received signal %s - finishing then exiting", signum)
        stop["flag"] = True

    # Signal handlers can only be registered from the main thread; guard so the
    # worker also runs when embedded (e.g. tests) in a non-main thread.
    try:
        signal.signal(signal.SIGTERM, _graceful)
        signal.signal(signal.SIGINT, _graceful)
    except ValueError:
        log.warning("not on main thread - skipping signal handlers")

    while not stop["flag"]:
        t0 = time.time()
        try:
            last_status = loop.run_cycle()
        except Exception:  # one bad cycle must not kill the worker
            log.exception("cycle failed")
            last_status = {"status": "error", "cycle": loop.cycle}
        # Sleep the remainder of the cadence (interruptibly).
        elapsed = time.time() - t0
        remaining = max(0.0, cfg.cycle_seconds - elapsed)
        slept = 0.0
        while slept < remaining and not stop["flag"]:
            time.sleep(min(1.0, remaining - slept))
            slept += 1.0

    log.info("worker stopped cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
