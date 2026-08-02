from __future__ import annotations

from ..config import Config
from .base import VectorStore
from .postgres import PostgresStore


def make_store(config: Config) -> VectorStore:
    kind = config.vector_store.lower()
    if kind == "postgres":
        return PostgresStore(dsn=config.pg_dsn)
    raise ValueError(f"Unknown VECTOR_STORE: {config.vector_store!r}")