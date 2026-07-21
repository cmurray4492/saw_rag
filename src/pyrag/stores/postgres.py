from __future__ import annotations
from typing import Any
from pgvector.psycopg import register_vector

import json
import psycopg

from .base import SearchHit, StoredChunk, VectorStore


class PostgresStore(VectorStore):
    def __init__(self, dsn:str) -> None:
        self._dsn = dsn
        self._conn: psycopg.Connection[Any] | None = None

    def _connect(self) -> psycopg.Connection[Any]:
        if self._conn is None or self._conn.closed:
            conn = psycopg.connect(self._dsn, autocommit=False)
            register_vector(conn)
            self._conn = conn

        return self._conn
