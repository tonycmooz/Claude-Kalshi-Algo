"""High-level persistence API used by the worker loop.

Wraps the SQLAlchemy Core schema with intention-revealing methods and keeps all
JSON (de)serialization and cross-dialect upsert logic in one place.
"""
from __future__ import annotations

import json
import time
from typing import Any, Optional

from sqlalchemy import and_, func, select
from sqlalchemy.engine import Engine

from ..data.types import MarketSnapshot
from . import db


class MarketStore:
    def __init__(self, engine: Engine):
        self.engine = engine

    # -- snapshots ----------------------------------------------------------
    def save_snapshots(self, snaps: list[MarketSnapshot]) -> int:
        """Insert snapshots, skipping (ticker, observed_ts) we already have."""
        if not snaps:
            return 0
        keys = {(s.ticker, round(s.observed_ts, 3)) for s in snaps}
        with self.engine.begin() as conn:
            existing = set()
            # Chunk the IN-clause to stay well under driver limits.
            tickers = list({s.ticker for s in snaps})
            for i in range(0, len(tickers), 500):
                rows = conn.execute(
                    select(db.snapshots.c.ticker, db.snapshots.c.observed_ts)
                    .where(db.snapshots.c.ticker.in_(tickers[i:i + 500]))).all()
                existing.update((r.ticker, round(r.observed_ts, 3)) for r in rows)
            new = [s for s in snaps if (s.ticker, round(s.observed_ts, 3)) not in existing]
            if not new:
                return 0
            conn.execute(db.snapshots.insert(), [self._snap_row(s) for s in new])
            return len(new)

    @staticmethod
    def _snap_row(s: MarketSnapshot) -> dict[str, Any]:
        return {
            "ticker": s.ticker, "observed_ts": s.observed_ts,
            "category": s.category, "close_ts": s.close_ts,
            "yes_bid": s.yes_bid, "yes_ask": s.yes_ask, "last_price": s.last_price,
            "volume": s.volume, "open_interest": s.open_interest,
            "price_history": json.dumps(list(s.price_history)),
            "extra": json.dumps(s.extra or {}),
        }

    # -- resolutions --------------------------------------------------------
    def save_resolution(self, ticker: str, result: int, ts: Optional[float] = None) -> None:
        ts = ts if ts is not None else time.time()
        with self.engine.begin() as conn:
            exists = conn.execute(select(db.resolutions.c.ticker)
                                  .where(db.resolutions.c.ticker == ticker)).first()
            if exists:
                conn.execute(db.resolutions.update()
                             .where(db.resolutions.c.ticker == ticker)
                             .values(result=int(result), resolved_ts=ts))
            else:
                conn.execute(db.resolutions.insert().values(
                    ticker=ticker, result=int(result), resolved_ts=ts))

    def count_labels(self) -> int:
        with self.engine.begin() as conn:
            return int(conn.execute(
                select(func.count()).select_from(db.resolutions)).scalar() or 0)

    def fetch_training_examples(self, limit: Optional[int] = None
                                ) -> list[MarketSnapshot]:
        """Return labelled snapshots: the earliest snapshot per resolved ticker.

        Using the earliest observed snapshot models the realistic decision point
        (we act on a market when we first see it), avoiding peeking at later,
        more-informed prices.
        """
        with self.engine.begin() as conn:
            j = db.snapshots.join(
                db.resolutions, db.snapshots.c.ticker == db.resolutions.c.ticker)
            rows = conn.execute(
                select(db.snapshots, db.resolutions.c.result)
                .select_from(j)
                .order_by(db.snapshots.c.ticker, db.snapshots.c.observed_ts)).all()
        # Keep the first row per ticker.
        out: dict[str, MarketSnapshot] = {}
        for r in rows:
            if r.ticker in out:
                continue
            out[r.ticker] = self._row_to_snapshot(r, result=int(r.result))
        snaps = sorted(out.values(), key=lambda s: s.observed_ts)
        return snaps[-limit:] if limit else snaps

    @staticmethod
    def _row_to_snapshot(r: Any, result: Optional[int]) -> MarketSnapshot:
        return MarketSnapshot(
            ticker=r.ticker, category=r.category or "unknown",
            observed_ts=r.observed_ts, close_ts=r.close_ts or r.observed_ts,
            yes_bid=r.yes_bid or 0.0, yes_ask=r.yes_ask or 0.0,
            last_price=r.last_price or 0.0, volume=r.volume or 0.0,
            open_interest=r.open_interest or 0.0,
            price_history=tuple(json.loads(r.price_history or "[]")),
            result=result, extra=json.loads(r.extra or "{}"),
        )

    # -- trades -------------------------------------------------------------
    def record_trade(self, *, ticker: str, side: str, contracts: int,
                     entry_price: float, fee: float, live: bool,
                     model_id: str | None) -> int:
        with self.engine.begin() as conn:
            res = conn.execute(db.trades.insert().values(
                ticker=ticker, side=side, contracts=contracts,
                entry_price=entry_price, fee=fee, opened_ts=time.time(),
                status="open", live=live, model_id=model_id))
            return int(res.inserted_primary_key[0])

    def open_trades(self) -> list[dict[str, Any]]:
        with self.engine.begin() as conn:
            rows = conn.execute(select(db.trades)
                                .where(db.trades.c.status == "open")).mappings().all()
            return [dict(r) for r in rows]

    def settle_trade(self, trade_id: int, result: int, pnl: float) -> None:
        with self.engine.begin() as conn:
            conn.execute(db.trades.update()
                         .where(db.trades.c.id == trade_id)
                         .values(status="settled", result=int(result),
                                 pnl=float(pnl), settled_ts=time.time()))

    def has_open_trade(self, ticker: str) -> bool:
        with self.engine.begin() as conn:
            return conn.execute(
                select(db.trades.c.id).where(and_(
                    db.trades.c.ticker == ticker,
                    db.trades.c.status == "open"))).first() is not None

    # -- key/value state ----------------------------------------------------
    def get_state(self, key: str, default: Any = None) -> Any:
        with self.engine.begin() as conn:
            row = conn.execute(select(db.state.c.value)
                               .where(db.state.c.key == key)).first()
        return json.loads(row.value) if row else default

    def set_state(self, key: str, value: Any) -> None:
        payload = json.dumps(value)
        with self.engine.begin() as conn:
            exists = conn.execute(select(db.state.c.key)
                                  .where(db.state.c.key == key)).first()
            if exists:
                conn.execute(db.state.update().where(db.state.c.key == key)
                             .values(value=payload, updated_ts=time.time()))
            else:
                conn.execute(db.state.insert().values(
                    key=key, value=payload, updated_ts=time.time()))

    # -- equity time-series -------------------------------------------------
    def record_equity(self, bankroll: float, realized_pnl: float,
                      open_positions: int, live: bool) -> None:
        with self.engine.begin() as conn:
            conn.execute(db.equity.insert().values(
                ts=time.time(), bankroll=bankroll, realized_pnl=realized_pnl,
                open_positions=open_positions, live=live))

    # -- model registry rows -----------------------------------------------
    def insert_model(self, *, model_id: str, score: float, config_json: str,
                     validation_json: str, test_json: str, n_train: int,
                     active: bool) -> None:
        with self.engine.begin() as conn:
            conn.execute(db.models.insert().values(
                id=model_id, created_ts=time.time(), score=score,
                config_json=config_json, validation_json=validation_json,
                test_json=test_json, n_train=n_train, active=active))

    def set_active_model(self, model_id: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(db.models.update().values(active=False))
            conn.execute(db.models.update()
                         .where(db.models.c.id == model_id).values(active=True))

    def get_active_model(self) -> Optional[dict[str, Any]]:
        with self.engine.begin() as conn:
            row = conn.execute(select(db.models)
                               .where(db.models.c.active.is_(True))
                               .order_by(db.models.c.created_ts.desc())).mappings().first()
            return dict(row) if row else None
