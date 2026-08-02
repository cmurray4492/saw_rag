from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()

def _env(key: str, default: str) -> str:
    val = os.getenv(key)
    return val if val is not None and val != "" else default

def _env_int(key: str, default: int) -> int:
    return int(_env(key, str(default)))


@dataclass(frozen=True)
class Config:
    openai_base_url: str | None
    openai_api_key: str
    chat_model: str
    vector_store: str
    pg_host: str
    pg_port: int
    pg_user: str
    pg_password: str
    pg_db: str

    @property
    def pg_dsn(self) -> str:
        return (
            f"postgresql://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_db}"
        )


def load_config() -> Config:
    base_url_raw = os.getenv("OPENAI_BASE_URL", "").strip()
    return Config(
        openai_base_url=base_url_raw or None,
        openai_api_key=_env("OPENAI_API_KEY", "unused-local"),
        chat_model=_env("CHAT_MODEL", "gemma3:latest"),
        vector_store=_env("VECTOR_STORE", "postgres"),
        pg_host=_env("POSTGRES_HOST", "localhost"),
        pg_port=_env_int("POSTGRES_PORT", 5432),
        pg_user=_env("POSTGRES_USER", "pyrag"),
        pg_password=_env("POSTGRES_PASSWORD", "pyrag"),
        pg_db=_env("POSTGRES_DB", "pyrag"),
    )