from __future__ import annotations

import base64
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from openai import OpenAI

if TYPE_CHECKING:
    from .config import Config

_PORTABLE_IMAGE_SUFFIXES = {".png", ".jpg", "jpeg"}
_MIME_BY_SUFFIX = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


@contextmanager
def _as_portable_image(path: Path) -> Iterator[Path]:
    if path.suffix.lower() in _PORTABLE_IMAGE_SUFFIXES:
        yield path
        return

    from PIL import Image

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(path)
    try:
        with Image.open(path)


class Message(TypedDict):
    role: str
    content: str


class ChatClient:
    def __init__(self, base_url: str | None, api_key: str, model: str) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    @classmethod
    def from_config(cls, cfg: Config) -> ChatClient:
        return cls(cfg.openai_base_url, cfg.openai_api_key, cfg.chat_model)

    def stream(self, messages: list[Message]) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            if not chunk.choices:
                continue
            piece = chunk.choices[0].delta.content
            if piece:
                yield piece
