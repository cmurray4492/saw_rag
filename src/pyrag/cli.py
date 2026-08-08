from __future__ import annotations

import logging
import typer

from pathlib import Path

from .config import load_config
from .embeddings import Embedder
from .ingest import Ingestor
from .llm import ChatClient
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


@app.command("chat")
def chat_cmd() -> None:
    cfg = load_config()
    chat = ChatClient.from_config(cfg)


    typer.echo(
        f"pyrag chat - model={cfg.chat_model}\n"
        "Type your question. Commands: /reset to clear history, /exit to quit."
    )

    messages: list = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            question = typer.prompt("\nyou", prompt_suffix="> ")
        except (EOFError, KeyboardInterrupt):
            type.echo("")
            break

        q = question.strip()
        if not q:
            continue
        if q in {"/exit", "/quit"}:
            break
        if q == "/reset":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            typer.echo("(history deleted)")
            continue

        messages.append({"role":"user", "content": q})

        typer.echo("\nassistant> ", nl=False)
        answer_parts: list[str] = []
        for piece in chat.stream(messages):
            typer.echo(piece, nl=False)
            answer_parts.append(piece)
        typer.echo("")
        messages.append({"role":"assistant", "content":"".join(answer_parts)})


if __name__ == "__main__":
    app()
