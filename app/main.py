import logging
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import UUID, uuid4
from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from app.config import settings
from app.rag import InMemoryRagStore
from app.rate_limit import RedisRateLimiter
from app.schemas import (AskRequest, AskResponse, DocumentCreate, EvaluationDatasetCreate,
                         SearchRequest, SearchResult)
from rag_core.llm import OllamaClient
from rag_core.persistent_rag import PersistentRagService
from rag_core.parsing import parse_upload

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

rate_limiter = RedisRateLimiter(settings.redis_url, settings.rate_limit_per_minute,
                                settings.rate_limit_fail_open,
                                trusted_api_keys=set(settings.tenant_api_keys or {}) if settings.auth_required else set())


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        yield
    finally:
        await rate_limiter.close()


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.middleware("http")(rate_limiter.dispatch)


@app.middleware("http")
async def request_telemetry(request, call_next):
    request_id = str(uuid4())
    started = perf_counter()
    try:
        response = await call_next(request)
    except Exception as error:
        logger.error("http_request request_id=%s method=%s status_code=500 duration_ms=%.2f error_type=%s",
                     request_id, request.method, (perf_counter() - started) * 1000,
                     type(error).__name__)
        raise
    response.headers["X-Request-ID"] = request_id
    logger.info("http_request request_id=%s method=%s status_code=%s duration_ms=%.2f",
                request_id, request.method, response.status_code, (perf_counter() - started) * 1000)
    return response


store = InMemoryRagStore(settings.chunk_size, settings.chunk_overlap)
_persistent_store = (PersistentRagService(settings.database_url, settings.qdrant_url, settings.qdrant_collection,
                     settings.embedding_dimensions, settings.chunk_size, settings.chunk_overlap,
                     settings.embedding_provider, settings.ollama_url, settings.ollama_embedding_model,
                     settings.redis_url, settings.qdrant_api_key)
                     if settings.rag_storage_backend == "persistent" else None)
ollama = OllamaClient(settings.ollama_url, settings.ollama_chat_model)


def _tenant_id(x_api_key: str | None) -> str:
    try:
        return settings.tenant_for_key(x_api_key)
    except PermissionError as error:
        raise HTTPException(status_code=401, detail=str(error)) from error


def _tenant_collection(collection: str, x_api_key: str | None) -> str:
    """Create an in-memory namespace for the authenticated tenant and collection."""
    tenant_id = _tenant_id(x_api_key)
    return f"{tenant_id}--{collection}"


@app.get("/health")
def health() -> dict[str, object]:
    rag = store.metrics() if _persistent_store is None else {"backend": "postgresql_qdrant",
          "collection": settings.qdrant_collection, "embedding_model": _persistent_store.embedder.model_name}
    return {"status": "ok", "rag": rag}


@app.get("/readyz")
def ready() -> JSONResponse:
    """Report readiness only after required storage services respond."""
    dependencies = ({"in_memory": True} if _persistent_store is None else _persistent_store.readiness())
    ready_status = all(dependencies.values())
    return JSONResponse(status_code=200 if ready_status else 503,
                        content={"status": "ready" if ready_status else "not_ready",
                                 "storage_backend": settings.rag_storage_backend, "dependencies": dependencies})


@app.post("/documents", status_code=201)
def create_document(document: DocumentCreate, x_api_key: str | None = Header(default=None)) -> dict[str, object]:
    collection = _tenant_collection(document.collection, x_api_key)
    if _persistent_store is not None:
        result = _persistent_store.ingest_text(_tenant_id(x_api_key), document.collection,
                                                document.source or document.title, document.content, document.metadata)
        logger.info("persisted document_id=%s chunks=%s", result["document_id"], result["chunks_created"])
        return result
    result = store.ingest(document.title, document.content, document.source, collection, document.metadata)
    logger.info("ingested document_id=%s chunks=%s", result["document_id"], result["chunks_created"])
    return result


@app.get("/documents")
def list_documents(collection: str = Query(default="default", pattern=r"^[a-zA-Z0-9_-]{1,64}$"),
                    limit: int = Query(default=100, ge=1, le=500), offset: int = Query(default=0, ge=0, le=1_000_000),
                    x_api_key: str | None = Header(default=None)) -> list[dict[str, object]]:
    if _persistent_store is not None:
        return _persistent_store.list_documents(_tenant_id(x_api_key), collection, limit, offset)
    return store.documents(_tenant_collection(collection, x_api_key), limit, offset)


@app.post("/documents/upload", status_code=201)
async def upload_document(file: UploadFile = File(...),
                          collection: str = Query(default="default", pattern=r"^[a-zA-Z0-9_-]{1,64}$"),
                          x_api_key: str | None = Header(default=None)) -> dict[str, object]:
    raw = await file.read(settings.max_upload_bytes + 1)
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Uploaded file exceeds MAX_UPLOAD_BYTES")
    try:
        content, suffix = parse_upload(file.filename or "upload.txt", file.content_type or "text/plain", raw)
    except (ValueError, UnicodeDecodeError) as error:
        raise HTTPException(status_code=415, detail=str(error)) from error
    except ImportError as error:
        raise HTTPException(status_code=503, detail="The parser dependency for this file type is not installed") from error
    if not content.strip():
        raise HTTPException(status_code=422, detail="Uploaded file contains no extractable text")
    tenant = _tenant_id(x_api_key)
    title = file.filename or "upload"
    metadata = {"filename": title, "extension": suffix}
    if _persistent_store is not None:
        return _persistent_store.ingest_text(tenant, collection, title, content, metadata)
    return store.ingest(title, content, title, _tenant_collection(collection, x_api_key), metadata)


@app.post("/jobs/ingest", status_code=202)
def enqueue_ingestion(document: DocumentCreate, x_api_key: str | None = Header(default=None)) -> dict[str, object]:
    if _persistent_store is None:
        raise HTTPException(status_code=503, detail="Background ingestion requires persistent storage")
    tenant = _tenant_id(x_api_key)
    source = document.source or document.title
    job = _persistent_store.create_ingest_job(tenant, document.collection, source)
    try:
        from workers.tasks import ingest_text
        ingest_text.delay(tenant, document.collection, source, document.content, document.metadata, job["job_id"])
    except Exception as error:
        _persistent_store.postgres.update_job(UUID(job["job_id"]), "FAILED", error="Unable to enqueue ingestion job", progress=0)
        logger.error("Unable to enqueue ingestion job error_type=%s", type(error).__name__)
        raise HTTPException(status_code=503, detail="Background worker queue is unavailable") from error
    return job


@app.get("/jobs/{job_id}")
def get_ingestion_job(job_id: str, x_api_key: str | None = Header(default=None)) -> dict[str, object]:
    if _persistent_store is None:
        raise HTTPException(status_code=503, detail="Job lookup requires persistent storage")
    tenant = _tenant_id(x_api_key)
    job = _persistent_store.get_job(tenant, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/evaluation/datasets", status_code=201)
def create_evaluation_dataset(dataset: EvaluationDatasetCreate,
                             x_api_key: str | None = Header(default=None)) -> dict[str, object]:
    if _persistent_store is None:
        raise HTTPException(status_code=503, detail="Persistent evaluation datasets require persistent storage")
    try:
        return _persistent_store.create_evaluation_dataset(
            _tenant_id(x_api_key), dataset.collection, dataset.name, dataset.description,
            [case.model_dump() for case in dataset.cases])
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/evaluation/datasets")
def list_evaluation_datasets(
        collection: str = Query(default="default", pattern=r"^[a-zA-Z0-9_-]{1,64}$"),
        limit: int = Query(default=100, ge=1, le=500), offset: int = Query(default=0, ge=0, le=1_000_000),
        x_api_key: str | None = Header(default=None)) -> list[dict[str, object]]:
    if _persistent_store is None:
        raise HTTPException(status_code=503, detail="Persistent evaluation datasets require persistent storage")
    return _persistent_store.list_evaluation_datasets(_tenant_id(x_api_key), collection, limit, offset)


@app.post("/evaluation/datasets/{dataset_id}/runs", status_code=201)
def run_evaluation_dataset(dataset_id: str,
                           collection: str = Query(default="default", pattern=r"^[a-zA-Z0-9_-]{1,64}$"),
                           top_k: int = Query(default=5, ge=1, le=20),
                           x_api_key: str | None = Header(default=None)) -> dict[str, object]:
    if _persistent_store is None:
        raise HTTPException(status_code=503, detail="Persistent evaluation runs require persistent storage")
    result = _persistent_store.run_evaluation_dataset(_tenant_id(x_api_key), collection, dataset_id, top_k)
    if result is None:
        raise HTTPException(status_code=404, detail="Evaluation dataset not found")
    return result


@app.get("/evaluation/datasets/{dataset_id}/runs")
def list_evaluation_runs(dataset_id: str,
                         collection: str = Query(default="default", pattern=r"^[a-zA-Z0-9_-]{1,64}$"),
                         limit: int = Query(default=100, ge=1, le=500), offset: int = Query(default=0, ge=0, le=1_000_000),
                         x_api_key: str | None = Header(default=None)) -> list[dict[str, object]]:
    if _persistent_store is None:
        raise HTTPException(status_code=503, detail="Persistent evaluation runs require persistent storage")
    tenant = _tenant_id(x_api_key)
    if _persistent_store.get_evaluation_dataset(tenant, collection, dataset_id) is None:
        raise HTTPException(status_code=404, detail="Evaluation dataset not found")
    return _persistent_store.list_evaluation_runs(tenant, collection, dataset_id, limit, offset)


@app.post("/search", response_model=list[SearchResult])
def search(request: SearchRequest, x_api_key: str | None = Header(default=None)) -> list[dict[str, object]]:
    collection = _tenant_collection(request.collection, x_api_key)
    if _persistent_store is not None:
        return _persistent_store.search(_tenant_id(x_api_key), request.collection,
                                        request.query, request.limit, request.filters)
    return store.search(request.query, request.limit, collection, request.filters)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, x_api_key: str | None = Header(default=None)) -> dict[str, object]:
    """Retrieve tenant-scoped context and generate a grounded answer through local Ollama."""
    collection = _tenant_collection(request.collection, x_api_key)
    if _persistent_store is not None:
        sources = _persistent_store.search(_tenant_id(x_api_key), request.collection,
                                           request.query, request.limit, request.filters)
    else:
        sources = store.search(request.query, request.limit, collection, request.filters)
    try:
        answer = ollama.answer(request.query, sources, request.model)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"answer": answer, "sources": sources}
