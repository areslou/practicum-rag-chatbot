"""Settings, read once from environment variables (or a local .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Folders scanned for documents. data/sample ships with the repo;
    # data/private is git-ignored for files you should not publish.
    data_dirs: tuple[Path, ...] = field(
        default_factory=lambda: tuple(
            Path(p) if Path(p).is_absolute() else ROOT / p
            for p in os.getenv("RAG_DATA_DIRS", "data/sample,data/private").split(",")
            if p.strip()
        )
    )
    storage_dir: Path = field(
        default_factory=lambda: ROOT / os.getenv("RAG_STORAGE_DIR", "storage")
    )
    collection: str = field(default_factory=lambda: os.getenv("RAG_COLLECTION", "practicum"))

    # Models. Leave RAG_LLM_MODEL empty to use the library's current default Gemini model.
    llm_model: str = field(default_factory=lambda: os.getenv("RAG_LLM_MODEL", ""))
    # Model used to grade answers in evals. Empty = same as llm_model. Using a different model
    # spreads the work across two free-tier daily quotas.
    judge_model: str = field(default_factory=lambda: os.getenv("RAG_JUDGE_MODEL", ""))
    embed_model: str = field(
        default_factory=lambda: os.getenv("RAG_EMBED_MODEL", "gemini-embedding-001")
    )
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))

    # Offline mode swaps Gemini for deterministic mock models (used by tests and CI).
    offline: bool = field(default_factory=lambda: _env_bool("RAG_OFFLINE"))

    # Retrieval and chunking.
    chunk_size: int = field(default_factory=lambda: int(os.getenv("RAG_CHUNK_SIZE", "256")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("RAG_CHUNK_OVERLAP", "32")))
    top_k: int = field(default_factory=lambda: int(os.getenv("RAG_TOP_K", "3")))
    # Chunks scoring below this similarity are dropped. 0 disables the filter.
    min_score: float = field(default_factory=lambda: float(os.getenv("RAG_MIN_SCORE", "0")))


def get_settings() -> Settings:
    return Settings()
