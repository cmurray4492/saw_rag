from __future__ import annotations
from .config import load_config
from .llm import ChatClient

import typer

app = typer.Typer(add_completion=False, help="Local RAG CLI")

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the users questions clearly"
    "and concisely."
)

@app.command("chat")
def chat_cmd() -> None:
    cfg = load_config()
    chat = ChatClient.from_config(cfg)

    typer.echo(
        f"pyrag chat - model={cfg.chat_model}\n"
        f"Type our question, Commands: /reset to clear history, /exit to exit"
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            question = typer.prompt("\nYou", prompt_suffix="> ")
        except (EOFError, KeyboardInterrupt):
            typer.echo("\nExiting...")
            break

        q = question.strip()
        if not q:
            continue
        if q in ("/exit", "/quit"):
            break
        if q == "/reset":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            typer.echo("Conversation history cleared.")
            continue

        messages.append({"role": "user", "content": q})

        typer.echo("\nAssistant: ", nl=False)
        answer_parts: list[str] = []
        for piece in chat.stream(messages):
            typer.echo(piece, nl=False)
            answer_parts.append(piece)
        typer.echo("")
        messages.append({"role": "assistant", "content": "".join(answer_parts)})


if __name__ == "__main__":
    app()
