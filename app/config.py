from dataclasses import dataclass, field
import json
import os


def _boolean_environment(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean (true/false, 1/0, yes/no, or on/off)")


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "MCP RAG Engineer")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "80"))
    auth_required: bool = field(default_factory=lambda: _boolean_environment("AUTH_REQUIRED", False))
    rag_storage_backend: str = os.getenv("RAG_STORAGE_BACKEND", "in_memory")
    database_url: str = os.getenv("DATABASE_URL", "")
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "rag_chunks")
    embedding_dimensions: int = int(os.getenv("EMBEDDING_DIMENSIONS", "384"))
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "0"))
    rate_limit_fail_open: bool = field(default_factory=lambda: _boolean_environment("RATE_LIMIT_FAIL_OPEN", False))
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "hash")
    ollama_embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", "52428800"))
    tenant_api_keys: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.qdrant_api_key == "":
            object.__setattr__(self, "qdrant_api_key", None)
        raw_keys = os.getenv("TENANT_API_KEYS", "{}") or "{}"
        try:
            keys = json.loads(raw_keys)
        except json.JSONDecodeError as error:
            raise ValueError("TENANT_API_KEYS must be a JSON object mapping API keys to tenant IDs") from error
        if not isinstance(keys, dict) or any(not isinstance(key, str) or not isinstance(value, str)
                                             or not key or not value for key, value in keys.items()):
            raise ValueError("TENANT_API_KEYS must map non-empty string API keys to non-empty tenant IDs")
        object.__setattr__(self, "tenant_api_keys", keys)
        if self.auth_required and not keys:
            raise ValueError("AUTH_REQUIRED=true requires at least one TENANT_API_KEYS entry")
        if self.rag_storage_backend not in {"in_memory", "persistent"}:
            raise ValueError("RAG_STORAGE_BACKEND must be in_memory or persistent")
        if self.rag_storage_backend == "persistent" and not self.database_url:
            raise ValueError("DATABASE_URL is required when RAG_STORAGE_BACKEND=persistent")
        if self.embedding_provider not in {"hash", "ollama"}:
            raise ValueError("EMBEDDING_PROVIDER must be hash or ollama")
        if self.chunk_size < 1 or self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_SIZE must be positive and CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.max_upload_bytes < 1 or self.embedding_dimensions < 1:
            raise ValueError("MAX_UPLOAD_BYTES and EMBEDDING_DIMENSIONS must be positive")
        if self.rate_limit_per_minute < 0:
            raise ValueError("RATE_LIMIT_PER_MINUTE cannot be negative")

    def tenant_for_key(self, api_key: str | None) -> str:
        if not self.auth_required:
            return "local-dev"
        if not api_key or api_key not in (self.tenant_api_keys or {}):
            raise PermissionError("A valid X-API-Key is required")
        return str(self.tenant_api_keys[api_key])


settings = Settings()
