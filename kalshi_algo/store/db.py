"""Database schema (SQLAlchemy Core).

Core (not the ORM) is used deliberately: the schema is small and explicit, and
the same table definitions run unchanged on Railway Postgres and on a local
SQLite file, which lets the entire worker loop be exercised offline in tests.
JSON-ish columns are stored as TEXT (``json.dumps``) for cross-dialect parity.
"""
from __future__ import annotations

from sqlalchemy import (
    Boolean, Column, Float, Integer, MetaData, String, Table, Text,
    UniqueConstraint, create_engine,
)
from sqlalchemy.engine import Engine

metadata = MetaData()

# Every observed open-market snapshot (the raw decision-time inputs).
snapshots = Table(
    "snapshots", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ticker", String(128), nullable=False, index=True),
    Column("observed_ts", Float, nullable=False, index=True),
    Column("category", String(64)),
    Column("close_ts", Float),
    Column("yes_bid", Float), Column("yes_ask", Float), Column("last_price", Float),
    Column("volume", Float), Column("open_interest", Float),
    Column("price_history", Text), Column("extra", Text),
    UniqueConstraint("ticker", "observed_ts", name="uq_snapshot_ticker_ts"),
)

# Resolutions (labels) arrive later, once a market settles.
resolutions = Table(
    "resolutions", metadata,
    Column("ticker", String(128), primary_key=True),
    Column("result", Integer, nullable=False),
    Column("resolved_ts", Float, nullable=False),
)

# Orders we placed (paper or live) and their realized outcome.
trades = Table(
    "trades", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ticker", String(128), nullable=False, index=True),
    Column("side", String(8)), Column("contracts", Integer),
    Column("entry_price", Float), Column("fee", Float),
    Column("opened_ts", Float, index=True),
    Column("status", String(16), default="open", index=True),  # open|settled
    Column("result", Integer), Column("pnl", Float),
    Column("settled_ts", Float),
    Column("live", Boolean, default=False),
    Column("model_id", String(64)),
)

# Trained model registry (blobs live on the volume; metrics live here).
models = Table(
    "models", metadata,
    Column("id", String(64), primary_key=True),
    Column("created_ts", Float, nullable=False),
    Column("score", Float),
    Column("config_json", Text),
    Column("validation_json", Text),
    Column("test_json", Text),
    Column("n_train", Integer),
    Column("active", Boolean, default=False, index=True),
)

# Simple key/value state (bankroll, daily PnL bucket, last cycle, ...).
state = Table(
    "state", metadata,
    Column("key", String(64), primary_key=True),
    Column("value", Text),
    Column("updated_ts", Float),
)

# Equity / metric time-series for monitoring.
equity = Table(
    "equity", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", Float, nullable=False, index=True),
    Column("bankroll", Float),
    Column("realized_pnl", Float),
    Column("open_positions", Integer),
    Column("live", Boolean),
)


def make_engine(database_url: str) -> Engine:
    """Create an engine and ensure the schema exists."""
    connect_args = {}
    if database_url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
    engine = create_engine(database_url, connect_args=connect_args,
                           pool_pre_ping=True, future=True)
    metadata.create_all(engine)
    return engine
