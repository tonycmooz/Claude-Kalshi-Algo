"""Runtime configuration assembled from environment variables.

This is the single place that reads the process environment, so the rest of the
code stays env-agnostic and testable.  Everything has a safe default; in
particular **trading is paper by default** and only flips to live when
``LIVE_TRADING=1`` *and* the account is configured - the "shadow then promote"
model chosen for deployment.
"""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass


def _b(name: str, default: bool = False) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in (
        "1", "true", "yes", "on")


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class RiskCaps:
    """Hard limits enforced before any live order is sent."""
    max_position_dollars: float = 50.0     # max $ risked on a single market
    max_total_exposure: float = 500.0      # max $ across all open positions
    max_open_positions: int = 40
    daily_loss_limit: float = 100.0        # halt live trading past this daily loss
    max_contracts_per_order: int = 250

    @classmethod
    def from_env(cls) -> "RiskCaps":
        return cls(
            max_position_dollars=_f("RISK_MAX_POSITION_DOLLARS", 50.0),
            max_total_exposure=_f("RISK_MAX_TOTAL_EXPOSURE", 500.0),
            max_open_positions=_i("RISK_MAX_OPEN_POSITIONS", 40),
            daily_loss_limit=_f("RISK_DAILY_LOSS_LIMIT", 100.0),
            max_contracts_per_order=_i("RISK_MAX_CONTRACTS_PER_ORDER", 250),
        )


@dataclass(frozen=True)
class RuntimeConfig:
    # --- data source -------------------------------------------------------
    data_source: str = "kalshi"            # "kalshi" | "sim"
    kalshi_base_url: str = "https://api.elections.kalshi.com/trade-api/v2"
    kalshi_key_id: str | None = None
    kalshi_private_key_pem: bytes | None = None

    # --- persistence -------------------------------------------------------
    database_url: str = "sqlite:////tmp/kalshi_algo.db"
    model_dir: str = "/data/models"

    # --- trading -----------------------------------------------------------
    live_trading: bool = False
    trading_enabled: bool = True           # global kill-switch
    starting_bankroll: float = 10_000.0
    risk: RiskCaps = RiskCaps()

    # --- loop cadence ------------------------------------------------------
    cycle_seconds: int = 300               # seconds between trading cycles
    learn_every_cycles: int = 12           # retrain/promote frequency (in cycles)
    min_labels_to_train: int = 800         # cold-start gate before first model

    # --- news / sentiment --------------------------------------------------
    news_api_url: str | None = None
    news_api_key: str | None = None

    # --- service -----------------------------------------------------------
    port: int = 8080

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        # Railway Postgres exposes DATABASE_URL as postgresql://...; SQLAlchemy
        # prefers the postgresql+psycopg2 dialect.
        db = os.getenv("DATABASE_URL", "sqlite:////tmp/kalshi_algo.db")
        if db.startswith("postgres://"):
            db = db.replace("postgres://", "postgresql+psycopg2://", 1)
        elif db.startswith("postgresql://"):
            db = db.replace("postgresql://", "postgresql+psycopg2://", 1)

        return cls(
            data_source=os.getenv("DATA_SOURCE", "kalshi").strip().lower(),
            kalshi_base_url=os.getenv(
                "KALSHI_BASE_URL",
                "https://api.elections.kalshi.com/trade-api/v2"),
            kalshi_key_id=os.getenv("KALSHI_API_KEY_ID") or None,
            kalshi_private_key_pem=_load_private_key(),
            database_url=db,
            model_dir=os.getenv("MODEL_DIR", "/data/models"),
            live_trading=_b("LIVE_TRADING", False),
            trading_enabled=_b("TRADING_ENABLED", True),
            starting_bankroll=_f("STARTING_BANKROLL", 10_000.0),
            risk=RiskCaps.from_env(),
            cycle_seconds=_i("CYCLE_SECONDS", 300),
            learn_every_cycles=_i("LEARN_EVERY_CYCLES", 12),
            min_labels_to_train=_i("MIN_LABELS_TO_TRAIN", 800),
            news_api_url=os.getenv("NEWS_API_URL") or None,
            news_api_key=os.getenv("NEWS_API_KEY") or None,
            port=_i("PORT", 8080),
        )


def _load_private_key() -> bytes | None:
    """Read the Kalshi RSA key from env, supporting raw PEM or base64-of-PEM.

    Storing multi-line PEM in env vars is awkward, so we also accept
    ``KALSHI_PRIVATE_KEY_B64`` (base64 of the PEM file).
    """
    raw = os.getenv("KALSHI_PRIVATE_KEY")
    if raw:
        return raw.encode("utf-8")
    b64 = os.getenv("KALSHI_PRIVATE_KEY_B64")
    if b64:
        try:
            return base64.b64decode(b64)
        except (ValueError, TypeError):
            return None
    return None
