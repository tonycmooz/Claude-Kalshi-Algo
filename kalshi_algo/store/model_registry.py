"""Model registry: persists trained Strategy blobs and tracks the champion.

Metrics and the active flag live in Postgres (via :class:`MarketStore`); the
pickled Strategy (predictor + feature builder + config) lives as a file on the
mounted volume (``MODEL_DIR``).  This split keeps the DB light and lets large
model artifacts survive redeploys on the volume.
"""
from __future__ import annotations

import json
import os
import pickle
import time
import uuid
from typing import Optional

from ..strategy import Strategy
from .market_store import MarketStore


class ModelRegistry:
    def __init__(self, store: MarketStore, model_dir: str):
        self.store = store
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def _path(self, model_id: str) -> str:
        return os.path.join(self.model_dir, f"{model_id}.pkl")

    def save(self, strategy: Strategy, *, score: float, validation: dict,
             test: dict, n_train: int, activate: bool = False) -> str:
        model_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        with open(self._path(model_id), "wb") as f:
            pickle.dump(strategy, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.store.insert_model(
            model_id=model_id, score=score,
            config_json=json.dumps(strategy.config.to_dict()),
            validation_json=json.dumps(validation),
            test_json=json.dumps(test), n_train=n_train, active=activate)
        if activate:
            self.store.set_active_model(model_id)
        return model_id

    def load_active(self) -> Optional[Strategy]:
        row = self.store.get_active_model()
        if not row:
            return None
        path = self._path(row["id"])
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def active_score(self) -> Optional[float]:
        row = self.store.get_active_model()
        return row["score"] if row else None

    def promote(self, model_id: str) -> None:
        self.store.set_active_model(model_id)
