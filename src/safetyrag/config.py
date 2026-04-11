"""Centralised configuration loaded from environment / .env file."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM
    groq_api_key: str = ""
    llm_model: str = "llama3-8b-8192"
    llm_temperature: float = 0.0

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Vector store
    vector_store_path: Path = Path("vector_store")

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval
    top_k: int = 4

    # Logging
    log_level: str = "INFO"


settings = Settings()
