"""
Application Configuration using Pydantic Settings
"""
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Application
    APP_NAME: str = "FinRAG Agent"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # API Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "mistral"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"

    # Vector Database
    CHROMA_PERSIST_DIR: str = "./data/vectors"
    CHROMA_COLLECTION_NAME: str = "financial_docs"

    # Document Storage
    PDF_UPLOAD_DIR: str = "./data/pdfs"
    MAX_FILE_SIZE_MB: int = 50

    # Text Splitting
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # RAG
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.3

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/finrag.db"

    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    @property
    def pdf_upload_path(self) -> Path:
        path = Path(self.PDF_UPLOAD_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def chroma_persist_path(self) -> Path:
        path = Path(self.CHROMA_PERSIST_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path


# Global settings instance
settings = Settings()
