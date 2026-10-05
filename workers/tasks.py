"""Background execution for persistent RAG jobs."""
from uuid import UUID

import httpx
from sqlalchemy.exc import OperationalError

from app.config import settings
from rag_core.errors import TransientDependencyError
from rag_core.persistent_rag import PersistentRagService
from workers.celery_app import celery_app


@celery_app.task(bind=True, name="rag.health_check")
def health_check(self) -> dict[str, str]:
    return {"task_id": self.request.id, "status": "ok"}


@celery_app.task(bind=True, name="rag.ingest_text",
                 autoretry_for=(ConnectionError, TimeoutError, OperationalError, httpx.TransportError,
                                TransientDependencyError),
                 retry_backoff=True, retry_jitter=True, max_retries=3)
def ingest_text(self, tenant_name: str, project_name: str, source: str, text: str,
                metadata: dict[str, object], job_id: str) -> dict[str, object]:
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL must be configured for persistent ingestion")
    service = PersistentRagService(settings.database_url, settings.qdrant_url, settings.qdrant_collection,
                                   settings.embedding_dimensions, settings.chunk_size, settings.chunk_overlap,
                                   settings.embedding_provider, settings.ollama_url, settings.ollama_embedding_model,
                                   settings.redis_url, settings.qdrant_api_key)
    return service.ingest_text(tenant_name, project_name, source, text, metadata, UUID(job_id))
