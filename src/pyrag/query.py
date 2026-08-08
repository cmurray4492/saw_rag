from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .embeddings import Embedder
from .stores.base import SearchHit, VectorStore
from .llm import Message

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about mythological creatures, "
    "folklore, mythology, and related topics. You have documents available to" 
    "you to use to answer questions. Use the context  and your general knowledge"
    "from outside the docuemnts to answer each question. Do not invent facts. " 
    "When you use a fact from the context, do NOT cite the source in parentheses."
    "Just write the response as part of the normal convesation. "
    "NEVER use phrasing like 'According to the documents... or anything similar. "
    "You ARE allowed to use your general knowledge about the world and mythology" 
    "provided the question is related to your area of expertise. Politely decline"
    "to answer questions not related to mythological creatures, folklore, mythology"
    "and related topics. "
    "keep your responses friendly and conversational. "
)


@dataclass
class RetrivedContext:
    hits: list[SearchHit]


def retrieve(
      store: VectorStore, embedder: Embedder, question: str, k: int
) -> RetrivedContext:
    [vec] = embedder.embed_text(question)
    return RetrivedContext(hits=store.search(question, vec, k))


def initial_messsages(system_prompt: str | None) -> list[Message]:
    base = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
    return [{"role": "system", "content": base}]
