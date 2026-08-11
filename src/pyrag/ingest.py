from __future__ import annotations
import hashlib
import logging 
import shutil
import threading
import time
from pathlib import Path
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from .chunking import chunk_text
from .config import Config
from .embeddings import Embedder
from .stores.base import StoredChunk, VectorStore

log = logging.getLogger(__name__)

TEXT_SUFFIXES = {".txt", ".md", ".markdown"}


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Ingestor:
    def __init__(self, config: Config, embedder: Embedder, store: VectorStore) -> None:
        self._config = config
        self._embedder = embedder
        self._store = store

    @property
    def store(self) -> VectorStore:
        return self._store

    @property
    def config(self) -> Config:
        return self._config

    def ingest_file(self, path: Path) -> None:
        suffix = path.suffix.lower()
        if suffix not in TEXT_SUFFIXES:
            log.info("Skipping unsupported file %s", path.name)
            return

        try:
            data = path.read_bytes()
        except FileNotFoundError:
            log.warning("File not found: %s", path)
            return

        content_hash = _hash_bytes(data)
        source_path = str(path.resolve())

        if self._store.has_document(source_path, content_hash):
            log.info("Unchanged, skipping embed %s", path.name)
            self._move_to_processed(path)
            return

        text = data.decode("utf-8", errors="replace")
        chunks = chunk_text(text, self._config.chunk_size, self._config.chunk_overlap)
        if not chunks:
            log.warning("No chunks generated for %s", path.name)
            self._move_to_processed(path)
            return

        log.info("Embedding %d chunks for %s", len(chunks), path.name)
        embeddings = self._embedder.embed([c.text for c in chunks])

        stored = [
            StoredChunk(
                index=c.index,
                text=c.text,
                embedding=emb,
                metadata={},
            )
            for c, emb in zip(chunks, embeddings, strict=True)
        ]

        self._store.upsert_document(source_path, content_hash, stored)
        log.info("Ingested %d chunks for %s", len(stored), path.name)
        self._move_to_processed(path)

    def _move_to_processed(self, path: Path) -> None:
        processed = self.config.processed_dir
        processed.mkdir(parents=True, exist_ok=True)
        target = processed / path.name
        if target.exists():
            stem, suffix = path.stem, path.suffix
            ts = time.strftime("%Y%m%d-%H%M%S")
            target = processed / f"{stem}.{ts}{suffix}"
        shutil.move(str(path), str(target))
        log.info("Moved -> %s", target.relative_to(self.documents_dir.parent()))
