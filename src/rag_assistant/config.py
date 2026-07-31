"""Typed, environment-driven configuration.

One immutable `Settings` object is built from the environment (plus an optional
`.env`) and threaded through the app. Nothing else reads `os.environ`, which
keeps components testable: build a `Settings` in a test and pass it in.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

_TRUTHY = {"1", "true", "yes", "on", "y"}


def _load_dotenv(path: Path) -> None:
    """Load `.env` if python-dotenv is available, else parse it ourselves."""
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(path, override=False)
        return
    except ImportError:
        pass
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _env(key: str, default: str | None = None) -> str | None:
    value = os.environ.get(key)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _env_int(key: str, default: int) -> int:
    raw = _env(key)
    try:
        return int(raw) if raw is not None else default
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = _env(key)
    try:
        return float(raw) if raw is not None else default
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = _env(key)
    return default if raw is None else raw.lower() in _TRUTHY


def _env_opt_int(key: str) -> int | None:
    raw = _env(key)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    provider: str = "huggingface"
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimensions: int | None = None
    batch_size: int = 64
    cache_size: int = 512

    @property
    def label(self) -> str:
        suffix = f"@{self.dimensions}d" if self.dimensions else ""
        model = "bag-of-words" if self.provider == "hashing" else self.model.rsplit("/", 1)[-1]
        return f"{self.provider}:{model}{suffix}"


@dataclass(frozen=True, slots=True)
class VectorStoreConfig:
    provider: str = "chroma"
    collection: str = "knowledge_base"
    directory: Path = Path("data/vectorstore")

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.collection}"


@dataclass(frozen=True, slots=True)
class LLMConfig:
    provider: str = "groq"
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.1
    max_tokens: int = 1024
    timeout: float = 60.0
    api_key: str | None = None

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    chunk_size: int = 900
    chunk_overlap: int = 150
    min_chunk_chars: int = 60

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    top_k: int = 5
    min_score: float = 0.15
    mmr_lambda: float = 0.7
    fetch_k_multiplier: int = 4
    max_context_chars: int = 12_000

    @property
    def fetch_k(self) -> int:
        return max(self.top_k, self.top_k * self.fetch_k_multiplier)


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    enable_llm_judge: bool = False
    relevance_threshold: float = 0.35
    faithfulness_threshold: float = 0.6


@dataclass(frozen=True)
class Settings:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    data_dir: Path = Path("data")
    log_level: str = "INFO"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def vector_dir(self) -> Path:
        return self.data_dir / "vectorstore" / self.vector_store.provider

    @property
    def db_path(self) -> Path:
        return self.data_dir / "knowledge_base.sqlite3"

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.upload_dir, self.vector_dir):
            path.mkdir(parents=True, exist_ok=True)

    def with_overrides(self, **kwargs: Any) -> Settings:
        """Return a copy with top-level sections replaced (used by benchmarks)."""
        return replace(self, **kwargs)

    @property
    def fingerprint(self) -> str:
        """Stable id for the retrieval configuration — groups benchmark runs."""
        return " | ".join(
            (
                self.embedding.label,
                self.vector_store.label,
                f"k={self.retrieval.top_k}",
                f"chunk={self.chunking.chunk_size}/{self.chunking.chunk_overlap}",
            )
        )

    @classmethod
    def from_env(cls, env_file: str | Path | None = ".env") -> Settings:
        if env_file:
            _load_dotenv(Path(env_file))

        llm_provider = (_env("LLM_PROVIDER", "groq") or "groq").lower()
        api_key_var = {
            "groq": "GROQ_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }.get(llm_provider)

        data_dir = Path(_env("DATA_DIR", "data") or "data").expanduser()
        vector_provider = (_env("VECTOR_STORE", "chroma") or "chroma").lower()

        embedding_defaults = EmbeddingConfig()
        llm_defaults = LLMConfig()

        settings = cls(
            embedding=EmbeddingConfig(
                provider=(_env("EMBEDDING_PROVIDER", "huggingface") or "").lower(),
                model=_env("EMBEDDING_MODEL", embedding_defaults.model) or embedding_defaults.model,
                dimensions=_env_opt_int("EMBEDDING_DIMENSIONS"),
                batch_size=_env_int("EMBEDDING_BATCH_SIZE", 64),
            ),
            vector_store=VectorStoreConfig(
                provider=vector_provider,
                collection=_env("VECTOR_COLLECTION", "knowledge_base") or "knowledge_base",
                directory=data_dir / "vectorstore" / vector_provider,
            ),
            llm=LLMConfig(
                provider=llm_provider,
                model=_env("LLM_MODEL", llm_defaults.model) or llm_defaults.model,
                temperature=_env_float("LLM_TEMPERATURE", 0.1),
                max_tokens=_env_int("LLM_MAX_TOKENS", 1024),
                timeout=_env_float("LLM_TIMEOUT", 60.0),
                api_key=_env(api_key_var) if api_key_var else None,
            ),
            chunking=ChunkingConfig(
                chunk_size=_env_int("CHUNK_SIZE", 900),
                chunk_overlap=_env_int("CHUNK_OVERLAP", 150),
            ),
            retrieval=RetrievalConfig(
                top_k=_env_int("TOP_K", 5),
                min_score=_env_float("MIN_SCORE", 0.15),
                mmr_lambda=_env_float("MMR_LAMBDA", 0.7),
                fetch_k_multiplier=_env_int("FETCH_K_MULTIPLIER", 4),
                max_context_chars=_env_int("MAX_CONTEXT_CHARS", 12_000),
            ),
            evaluation=EvaluationConfig(
                enable_llm_judge=_env_bool("ENABLE_LLM_JUDGE", False),
                relevance_threshold=_env_float("RELEVANCE_THRESHOLD", 0.35),
                faithfulness_threshold=_env_float("FAITHFULNESS_THRESHOLD", 0.6),
            ),
            data_dir=data_dir,
            log_level=(_env("LOG_LEVEL", "INFO") or "INFO").upper(),
        )
        return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    settings = Settings.from_env()
    settings.ensure_dirs()
    return settings
