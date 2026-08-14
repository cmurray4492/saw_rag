from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter, HybridFusion, MetadataQuery

from .base import SearchHit, StoredChunk, VectorStore


class WeaviateStore(VectorStore):

    P_SOURCE_PATH = "source_path"
    P_CONTENT_HASH = "content_hash"
    P_CHUNK_INDEX = "chunk_index"
    P_CONTENT = "content"
    P_CHUNK_METADATA = "chunk_metadata"
    P_DOCUMENT_METADATA = "document_metadata"
    P_INGESTED_AT = "ingested_at"
    P_CHUNK_COUNT = "chunk_count"

    def __init__(
            self,
            host: str,
            http_port: int,
            grpc_port: int,
            collection: str,
        ) -> None:
            self._host = host
            self._http_port = http_port
            self._grpc_port = grpc_port
            self._collection_name = collection
            self._client: weaviate.WeaviateClient | None = None

    def _connect(self) -> weaviate.WeaviateClient:
        if self._client is None or self._client.is_connected():
            self._client = weaviate.connect_to_local(
                 host=self._host,
                 port=self._http_port,
                 grpc_port=self._grpc_port,
            )
            self._ensure_collection()
        return self._client

    def _ensure_collection(self) -> None:
        assert self._client is not None
        if self._client.collections.exists(self._collection_name):
            return

        self._client.collections.create(
             name=self._collection_name,
             vector_config=Configure.Vectors.self_provided(),
             properties=[
                  Property(name=self.P_SOURCE_PATH, data_type=DataType.TEXT),
                  Property(name=self.P_CONTENT_HASH, data_type=DataType.TEXT),
                  Property(name=self.P_CHUNK_INDEX, data_type=DataType.INT),
                  Property(name=self.P_CONTENT, data_type=DataType.TEXT),
                  Property(name=self.P_CHUNK_METADATA, data_type=DataType.TEXT),
                  Property(name=self.P_DOCUMENT_METADATA, data_type=DataType.TEXT),
                  Property(name=self.P_INGESTED_AT, data_type=DataType.DATE),
                  Property(name=self.P_CHUNK_COUNT, data_type=DataType.INT),
             ],
        )
