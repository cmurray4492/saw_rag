from __future__ import annotations

import json
import logging
import threading

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .config import load_config
from .embeddings import Embedder
from .ingest import TEXT_SUFFIXES, Ingestor
from .llm import ChatClient, Message
from .query import build_user_message, initial_messages, retrieve
from .stores.factory import make_store

log = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatTurn] = []
    k: int | None = None


def create_app() -> FastAPI:
    cfg = load_config()
    cfg.ensure_dirs()

    store = make_store(cfg)
    embedder = Embedder.from_config(cfg)
    chat = ChatClient.from_config(cfg)
    ingestor = Ingestor(cfg, store, embedder)
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            store.close()

    app = FastAPI(title="pyrag", lifespan=lifespan)
    app.state.cfg = cfg
    app.state.store = store
    app.state.embedder = embedder
    app.state.chat = chat
    app.state.ingestor = ingestor
    app.state.lock = lock

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    return app


app = create_app()
