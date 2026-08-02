from __future__ import annotations

import typer

from .config import load_config
from .llm import ChatClient

app = typer.Typer(add_completion=False, help="Local RAG CLI.")

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's questions clearly "
    "and concisely."
)

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