from __future__ import annotations

import json
import psycopg

from typing import Any
from pgvector.psycopg import register_vector
from .base import SearchHit, StoredChunk, VectorStore


class PostgresStore(VectorStore):
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._conn: psycopg.Connection[Any] | None = None

    def _connect(self) -> psycopg.Connection[Any]:
        if self._conn is None or self._conn.closed:
            conn = psycopg.connect(self._dsn, autocommit=False)
            register_vector(conn)
            self._conn = conn

        return self._conn

    def has_document(self, source_path: str, content_hash: str) -> bool:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM documents WHERE source_path = %s AND content_hash = %s",
                (source_path, content_hash),
            )
            found = cur.fetchone() is not None
        conn.commit()
        return found

    def upsert_document(
            self,
            source_path: str,
            content_hash: str,
            chunks: list[StoredChunk],
    ) -> None:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents (source_path, content_hash, chunk_count)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (source_path) DO UPDATE
                        SET content_hash = EXCLUDED.content_hash,
                            chunk_count = EXCLUDED.chunk_count,
                            ingested_at = now()
                    returning id
                    """,
                    (source_path, content_hash, len(chunks))
                )
                doc_id = cur.fetchone()[0]

                cur.execute("delete from chunks where document_id = %s", (doc_id,))

                if chunks:
                    cur.executemany(
                        """
                        INSERT INTO chunks (document_id, chunk_index, content, embedding, metadata)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        [
                            (
                                doc_id,
                                c.index,
                                c.text,
                                c.embedding,
                                json.dump(c.metadata),
                            )
                            for c in chunks
                        ],
                    )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def delete_document(self, source_path: str) -> None:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM documents WHERE source_path = %s",
                (source_path,),
            )
            conn.commit()

    def search(
            self, query_text: str,
            query_embedding: list[float], k: int
    ) -> list[SearchHit]:
        """Retrieval is TODO"""
        raise NotImplementedError('search will be added later')

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
            self._conn = None
