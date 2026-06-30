from __future__ import annotations
from dataclasses import dataclass
from dotenv import load_dotenv

import os

load_dotenv()


def _env(key: str, default: str) -> str:
    val = os.getenv(key)
    return val if val is not None and val != "" else default


def _env_int(key: str, default: int) -> int:
    return int(_env(key , str(default)))   


@dataclass(frozen=True)
class Config:
    openai_base_url: str | None
    openai_api_key: str
    chat_model: str
    vector_store: str
    postgres_host: str
    postgres_port: int
    postgres_user: str
    postgres_password: str
    postgres_db: str

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


def load_config() -> Config:
    base_url_raw = os.getenv("OPENAI_BASE_URL", "").strip()
    return Config(
        openai_base_url=base_url_raw or None,
        openai_api_key=_env("OPENAI_API_KEY", "unused-local"),
        chat_model=_env("CHATGPT_MODEL", "gemma3:7b"),
        vector_store=_env("VECTOR_STORE", "postgres"),
        postgres_host=_env("POSTGRES_HOST", "localhost"),
        postgres_port=_env_int("POSTGRES_PORT", 5432),
        postgres_user=_env("POSTGRES_USER", "pyrag"),
        postgres_password=_env("POSTGRES_PASSWORD", "pyrag"),
        postgres_db=_env("POSTGRES_DB", "pyrag"),
    )





_________________
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=pyrag
POSTGRES_PASSWORD=pyrag
POSTGRES_DB=pyrag

