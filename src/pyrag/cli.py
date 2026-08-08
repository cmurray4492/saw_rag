from __future__ import annotations

import logging
import typer

from pathlib import Path

from .config import load_config
from .embeddings import Embedder
from .ingest import Ingestor
from .llm import ChatClient
from .query import build_user_message, initial_messages, retrieve
from .stores.factory import make_store


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _build_ingestor() -> Ingestor:
    cfg = load_config()
    cfg.ensure_dirs()
    store = make_store(cfg)
    embedder = Embedder.from_config(cfg)
    return Ingestor(cfg, embedder, store)


def _print_sources(hits: list) -> None:
    if not hits:
        return
    typer.echo("\nsources: ")
    for h in hits:
        name = Path(h.source_path).name
        typer.echo(
            f"   - {name} (chunk {h.chunk_index}), score {h.score:.2f}"
        )


app = typer.Typer(add_completion=False, help="Local RAG CLI.")

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's questions clearly "
    "and concisely."
)


@app.command("ingest")
def ingest_cmd(path: Path = typer.Argument(..., exists=True, readable=True)) -> None:
    _setup_logging()
    ingestor = _build_ingestor()
    try:
        ingestor.ingest_file(path)
    finally:
        ingestor.store.close()


@app.command("scan")
def scan_cmd() -> None:
    _setup_logging()
    ingestor = _build_ingestor()
    try:
        docs = ingestor.config.documents_dir
        for path in docs.iterdir():
            if path.is_file():
                ingestor.ingest_file(path)
    finally:
        ingestor.store.close()


@app.command("ask")
def ask_cmd(
    question: str = typer.Argument(..., help="One shot question to ask"),
    k: int = typer.Option(None, "--k", help="Override TOP_K"),
) -> None:
    _setup_logging()
    cfg = load_config()
    store = make_store(cfg)
    embedder = Embedder.from_config(cfg)
    chat = ChatClient.from_config(cfg)
    top_k = k if k is not None else cfg.top_k
    try:
        ctx = retrieve(store, embedder, question, top_k)
        messages = initial_messages(cfg.system_prompt)
        messages.append(
            {"role":"user", "content": build_user_message(question, ctx)}
        )
        for piece in chat.stream(messages):
            typer.echo(piece, nl=False)
        typer.echo("")
        _print_sources(ctx.hits)
    finally:
        store.close()


@app.command("chat")
def chat_cmd(
    k: int = typer.Option(None, "--k", help="Override TOP_K"),
) -> None:
    _setup_logging()
    cfg = load_config()
    store = make_store(cfg)
    embedder = Embedder.from_config(cfg)
    chat = ChatClient.from_config(cfg)
    top_k = k if k is not None else cfg.top_k

    typer.echo(
        f"pyrag chat - model={cfg.chat_model}\n"
        "Type your question. Commands: /reset to clear history, /exit to quit."
    )

    messages = initial_messages(cfg.system_prompt)
    try:
        while True:
            try:
                question = typer.prompt("\nyou", prompt_suffix="> ")
            except (EOFError, KeyboardInterrupt):
                typer.echo("")
                break

            q = question.strip()
            if not q:
                continue
            if q in {"/exit", "/quit"}:
                break
            if q == "/reset":
                messages = initial_messages(cfg.system_prompt)
                typer.echo("(history deleted)")
                continue

            ctx = retrieve(store, embedder, q, top_k)
            messages.append(
                {"role": "user", "content": build_user_message(q, ctx)}
            )

            typer.echo("\nassistant> ", nl=False)
            answer_parts: list[str] = []
            for piece in chat.stream(messages):
                typer.echo(piece, nl=False)
                answer_parts.append(piece)
            typer.echo("")
            messages.append({"role":"assistant", "content":"".join(answer_parts)})
            _print_sources(ctx.hits)
    finally:
        store.close()


if __name__ == "__main__":
    app()
