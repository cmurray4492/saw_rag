from __future__ import annotations
from dataclasses import dataclass
from dotenv import load_dotenv

import os

load_dotenv()


def _env(key: str, default: str) -> str:
    val = os.getenv(key)
    return val if val is not None and val != "" else default


@dataclass(frozen=True)
class Config:
    openai_base_url: str | None
    openai_api_key: str
    chat_model: str


def load_config() -> Config:
    base_url_raw = os.getenv("OPENAI_BASE_URL", "").strip()
    return Config(
        openai_base_url=base_url_raw or None,
        openai_api_key=_env("OPENAI_API_KEY", "unused-local"),
        chat_model=_env("CHATGPT_MODEL", "gemma3:7b"),
    )
