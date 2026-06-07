"""News / sentiment providers.

A pluggable signal source: given the markets we're about to evaluate, attach a
sentiment score (in ``[-1, 1]``) to each snapshot's ``extra['news_sentiment']``.
The feature builder picks this up automatically when present, so news improves
predictions live without changing the offline simulator pipeline.

* ``NullNewsProvider``  - default, no-op (used when no API is configured).
* ``HTTPNewsProvider``  - queries a configurable headlines API and scores
  sentiment with a small lexicon.  It is defensively coded: any failure yields a
  neutral 0.0 so the trading loop never breaks on a news outage.
"""
from __future__ import annotations

import logging
from typing import Protocol

from .types import MarketSnapshot

log = logging.getLogger(__name__)

_POSITIVE = {
    "surge", "soar", "gain", "rise", "beat", "win", "approve", "record",
    "boom", "rally", "growth", "up", "strong", "success", "expand", "lead",
}
_NEGATIVE = {
    "plunge", "drop", "fall", "miss", "lose", "reject", "cut", "crash",
    "weak", "decline", "down", "fail", "recession", "default", "ban", "fear",
}


class NewsProvider(Protocol):
    def enrich(self, snapshots: list[MarketSnapshot]) -> None: ...


class NullNewsProvider:
    def enrich(self, snapshots: list[MarketSnapshot]) -> None:
        return None


def _score_text(text: str) -> float:
    words = text.lower().replace(",", " ").replace(".", " ").split()
    if not words:
        return 0.0
    pos = sum(w in _POSITIVE for w in words)
    neg = sum(w in _NEGATIVE for w in words)
    if pos + neg == 0:
        return 0.0
    return max(-1.0, min(1.0, (pos - neg) / (pos + neg)))


class HTTPNewsProvider:
    """Generic headlines provider.

    ``api_url`` is expected to accept a ``q`` query param and return JSON with an
    ``articles`` list of objects having ``title`` (and optionally ``description``)
    - the shape of NewsAPI.org and many compatible services.  Adapt ``_query``
    for a different vendor.
    """

    def __init__(self, api_url: str, api_key: str | None = None,
                 timeout: float = 6.0, session=None):
        import requests
        self.api_url = api_url
        self.api_key = api_key
        self.timeout = timeout
        self.session = session or requests.Session()
        self._cache: dict[str, float] = {}

    def _query(self, term: str) -> float:
        if term in self._cache:
            return self._cache[term]
        try:
            params = {"q": term, "pageSize": 10, "language": "en"}
            headers = {"X-Api-Key": self.api_key} if self.api_key else {}
            r = self.session.get(self.api_url, params=params, headers=headers,
                                 timeout=self.timeout)
            r.raise_for_status()
            articles = r.json().get("articles", [])
            scores = [_score_text(
                f"{a.get('title', '')} {a.get('description', '')}")
                for a in articles]
            score = sum(scores) / len(scores) if scores else 0.0
        except Exception as exc:  # never let news break trading
            log.warning("news query failed for %r: %s", term, exc)
            score = 0.0
        self._cache[term] = score
        return score

    def enrich(self, snapshots: list[MarketSnapshot]) -> None:
        # Query once per category to stay within rate limits, then assign.
        terms = {s.category for s in snapshots if s.category}
        scores = {t: self._query(t) for t in terms}
        for s in snapshots:
            s.extra["news_sentiment"] = scores.get(s.category, 0.0)


def build_news_provider(api_url: str | None, api_key: str | None) -> NewsProvider:
    if api_url:
        return HTTPNewsProvider(api_url, api_key)
    return NullNewsProvider()
