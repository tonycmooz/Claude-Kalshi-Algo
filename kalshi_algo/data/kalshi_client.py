"""Live Kalshi REST client (Trade API v2).

This is the real-exchange adapter.  It is intentionally swappable with
:class:`~kalshi_algo.data.simulator.MarketSimulator` - both ultimately yield
:class:`MarketSnapshot` objects so the strategy code is agnostic to the source.

Authentication
--------------
Kalshi's v2 API authenticates each request with an API key id and an RSA private
key.  Every request is signed: the signature is ``RSA-PSS(SHA256)`` over the
string ``"{timestamp_ms}{METHOD}{path}"`` and sent in the ``KALSHI-ACCESS-*``
headers.  Public market-data endpoints work without auth.

Safety
------
``paper`` mode (the default) never submits real orders; ``create_order`` logs
the intended order and returns a simulated acknowledgement.  Set ``paper=False``
*and* supply credentials to trade live - a deliberately explicit opt-in.
"""
from __future__ import annotations

import base64
import logging
import time
from typing import Any, Iterable, Optional

import requests

from ..config import EXCLUDED_CATEGORIES
from .types import MarketSnapshot

log = logging.getLogger(__name__)

# Kalshi production base; the elections/demo hosts share the same path layout.
DEFAULT_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"


class KalshiAuth:
    """Signs requests with an RSA private key (PEM)."""

    def __init__(self, key_id: str, private_key_pem: bytes):
        # Imported lazily so the package works without cryptography installed
        # when only the simulator is used.
        from cryptography.hazmat.primitives import serialization

        self.key_id = key_id
        self._key = serialization.load_pem_private_key(private_key_pem, password=None)

    def headers(self, method: str, path: str) -> dict[str, str]:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        ts = str(int(time.time() * 1000))
        # Sign over timestamp + METHOD + path (path without query string).
        msg = (ts + method.upper() + path).encode("utf-8")
        signature = self._key.sign(
            msg,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
            "KALSHI-ACCESS-TIMESTAMP": ts,
        }


class KalshiClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        auth: Optional[KalshiAuth] = None,
        paper: bool = True,
        timeout: float = 10.0,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.auth = auth
        self.paper = paper
        self.timeout = timeout
        self.session = session or requests.Session()

    # -- low-level ----------------------------------------------------------
    def _request(self, method: str, path: str, *, params: Optional[dict] = None,
                 json: Optional[dict] = None) -> dict[str, Any]:
        url = self.base_url + path
        headers = {"Content-Type": "application/json"}
        if self.auth is not None:
            # The signed path must include the API prefix but not the host.
            signed_path = url.replace(self.base_url.rsplit("/trade-api", 1)[0], "")
            headers.update(self.auth.headers(method, signed_path))
        resp = self.session.request(
            method, url, params=params, json=json, headers=headers,
            timeout=self.timeout)
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    # -- public market data -------------------------------------------------
    def exchange_status(self) -> dict[str, Any]:
        return self._request("GET", "/exchange/status")

    def iter_markets(self, *, status: str = "open", limit: int = 100,
                     max_markets: int = 1000) -> Iterable[dict[str, Any]]:
        """Yield raw market dicts, transparently following pagination."""
        cursor = None
        fetched = 0
        while fetched < max_markets:
            params = {"status": status, "limit": min(limit, max_markets - fetched)}
            if cursor:
                params["cursor"] = cursor
            data = self._request("GET", "/markets", params=params)
            markets = data.get("markets", [])
            if not markets:
                break
            for m in markets:
                yield m
                fetched += 1
            cursor = data.get("cursor")
            if not cursor:
                break

    def orderbook(self, ticker: str, depth: int = 10) -> dict[str, Any]:
        return self._request("GET", f"/markets/{ticker}/orderbook",
                             params={"depth": depth})

    # -- account / trading --------------------------------------------------
    def balance(self) -> dict[str, Any]:
        return self._request("GET", "/portfolio/balance")

    def positions(self) -> dict[str, Any]:
        return self._request("GET", "/portfolio/positions")

    def create_order(self, ticker: str, side: str, action: str, count: int,
                     price_cents: int, *, order_type: str = "limit",
                     client_order_id: Optional[str] = None) -> dict[str, Any]:
        """Place an order.  In ``paper`` mode this is a no-op acknowledgement."""
        payload = {
            "ticker": ticker,
            "side": side,            # "yes" | "no"
            "action": action,        # "buy" | "sell"
            "count": int(count),
            "type": order_type,
            "yes_price" if side == "yes" else "no_price": int(price_cents),
            "client_order_id": client_order_id or f"algo-{int(time.time()*1000)}",
        }
        if self.paper or self.auth is None:
            log.info("[PAPER] order %s", payload)
            return {"status": "paper", "order": payload}
        return self._request("POST", "/portfolio/orders", json=payload)

    # -- adaptation to MarketSnapshot --------------------------------------
    @staticmethod
    def is_non_weather(market: dict[str, Any]) -> bool:
        blob = " ".join(str(market.get(k, "")) for k in
                        ("category", "series_ticker", "title", "subtitle")).lower()
        return not any(bad in blob for bad in EXCLUDED_CATEGORIES)

    def snapshot_open_markets(self, max_markets: int = 500
                              ) -> list[MarketSnapshot]:
        """Fetch open, non-weather markets and adapt to MarketSnapshot.

        ``result`` is left ``None`` because open markets are unresolved.
        """
        out: list[MarketSnapshot] = []
        now = time.time()
        for m in self.iter_markets(status="open", max_markets=max_markets):
            if not self.is_non_weather(m):
                continue
            try:
                out.append(self._to_snapshot(m, now))
            except (KeyError, TypeError, ValueError):
                continue
        return out

    @staticmethod
    def _to_snapshot(m: dict[str, Any], now: float) -> MarketSnapshot:
        c2d = lambda c: (c or 0) / 100.0  # cents -> dollars
        close_iso = m.get("close_time") or m.get("expiration_time")
        close_ts = _parse_ts(close_iso, default=now + 3600.0)
        return MarketSnapshot(
            ticker=m["ticker"],
            category=str(m.get("category", "unknown")).lower(),
            observed_ts=now,
            close_ts=close_ts,
            yes_bid=c2d(m.get("yes_bid")),
            yes_ask=c2d(m.get("yes_ask")),
            last_price=c2d(m.get("last_price")),
            volume=float(m.get("volume", 0) or 0),
            open_interest=float(m.get("open_interest", 0) or 0),
            price_history=(),
            result=None,
        )


def _parse_ts(value: Any, default: float) -> float:
    if not value:
        return default
    try:
        from datetime import datetime
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return default
