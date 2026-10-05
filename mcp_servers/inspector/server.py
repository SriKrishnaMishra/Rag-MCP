"""MCP tools for RAG inspection and explicitly confirmed project actions."""
from __future__ import annotations

import os
from pathlib import Path
import re

from mcp.server.fastmcp import FastMCP

from app.main import _persistent_store, app, store
from rag_core.evaluation import aggregate_case_metrics, score_case
from mcp_servers.inspector.approved_actions import (apply_approved_source_edit, inspect_docker_runtime,
                                                    preview_source_edit,
                                                    probe_repository_dependencies,
                                                    run_selected_repository_tests as run_tests)
from mcp_servers.inspector.connectors import inspect_repository_integrations

SERVER_ROOT = Path(__file__).resolve().parents[2]
# Point this at another repository when running the MCP server to inspect it.
# The MCP process can only read the repository explicitly selected at startup.
ROOT = Path(os.getenv("TARGET_REPOSITORY", str(SERVER_ROOT))).resolve()
mcp = FastMCP("RAG Backend Inspector")
SAFE_SUFFIXES = {".py", ".toml", ".md", ".yml", ".yaml", ".json", ".txt"}
INGEST_SUFFIXES = {".txt", ".md"}
IGNORED_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", "dist", "build"}
MAX_INGEST_CHARS = 5_000_000
MAX_QUERY_CHARS = 4_000
MAX_METADATA_ENTRIES = 32
MAX_SOURCE_READ_BYTES = 2_000_000
MAX_SEARCH_BYTES = 25_000_000
MAX_SEARCH_FILES = 5_000
MAX_ROUTE_RESULTS = 500
MAX_SCAN_DIRECTORIES = 20_000
MAX_PROJECT_FILES = 20_000


def _inside_selected_root(path: Path) -> bool:
    try:
        return path.resolve().is_relative_to(ROOT)
    except (OSError, RuntimeError):
        return False


def _bounded_repository_files(*, suffixes: set[str] | None = None,
                              names: set[str] | None = None,
                              max_files: int,
                              max_bytes: int | None,
                              max_file_bytes: int | None = None) -> tuple[list[Path], bool]:
    """Collect a deterministic, pruned repository file batch under explicit scan budgets."""
    files: list[Path] = []
    scanned_bytes = 0
    directories = 0
    truncated = False
    for directory, child_directories, filenames in os.walk(ROOT):
        directories += 1
        if directories > MAX_SCAN_DIRECTORIES:
            truncated = True
            break
        child_directories[:] = sorted(name for name in child_directories
                                       if not name.startswith(".") and name not in IGNORED_DIRS)
        for filename in sorted(filenames):
            if filename.startswith(".") or (suffixes is not None and Path(filename).suffix not in suffixes):
                continue
            if names is not None and filename not in names:
                continue
            path = Path(directory) / filename
            if not _inside_selected_root(path):
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if max_file_bytes is not None and size > max_file_bytes:
                continue
            if len(files) >= max_files or (max_bytes is not None and scanned_bytes + size > max_bytes):
                truncated = True
                break
            files.append(path)
            scanned_bytes += size
        if truncated:
            break
    return files, truncated

RAG_STACKS = {
    "starter": {
        "orchestration": "LlamaIndex or LangChain",
        "parsing": "pypdf for simple PDFs; Docling for complex PDFs",
        "vector_store": "Qdrant",
        "embeddings": "Sentence Transformers",
        "evaluation": "Ragas",
        "observability": "Langfuse",
    },
    "postgres": {
        "orchestration": "RAGLite or LlamaIndex",
        "parsing": "Docling",
        "vector_store": "PostgreSQL with pgvector",
        "embeddings": "FastEmbed or Sentence Transformers",
        "evaluation": "DeepEval or Ragas",
        "observability": "OpenLIT or Langfuse",
    },
    "enterprise": {
        "orchestration": "Haystack or LangGraph",
        "parsing": "Docling or Unstructured",
        "vector_store": "Qdrant, OpenSearch, or Milvus",
        "embeddings": "FastEmbed or FlagEmbedding",
        "evaluation": "Ragas and DeepEval in CI",
        "observability": "Langfuse or Arize Phoenix",
    },
}


def _safe_path(relative_path: str) -> Path:
    candidate = (ROOT / relative_path).resolve()
    if ROOT not in candidate.parents or candidate.suffix not in SAFE_SUFFIXES:
        raise ValueError("Only workspace-relative text source files may be read.")
    return candidate


def _validate_collection(collection: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", collection):
        raise ValueError("collection must contain 1-64 letters, digits, underscores, or hyphens")


def _validate_metadata(metadata: dict[str, object] | None) -> None:
    if metadata is None:
        return
    if not isinstance(metadata, dict) or len(metadata) > MAX_METADATA_ENTRIES:
        raise ValueError(f"metadata must be an object with at most {MAX_METADATA_ENTRIES} entries")
    for key, value in metadata.items():
        if not isinstance(key, str) or not 1 <= len(key) <= 128:
            raise ValueError("metadata keys must be strings of 1-128 characters")
        if not isinstance(value, str) or len(value) > 512:
            raise ValueError("metadata values must be strings of at most 512 characters")


def _validate_query(query: str) -> None:
    if not query.strip():
        raise ValueError("query must not be empty")
    if len(query) > MAX_QUERY_CHARS:
        raise ValueError(f"query must not exceed {MAX_QUERY_CHARS} characters")


@mcp.tool()
def inspect_project() -> dict[str, object]:
    """List source files and describe the current project."""
    paths, truncated = _bounded_repository_files(max_files=MAX_PROJECT_FILES, max_bytes=None)
    files = sorted(str(path.relative_to(ROOT)) for path in paths)
    return {"root": str(ROOT), "file_count": len(files), "files": files[:100],
            "scan_truncated": truncated, "mode": "read_only"}


def _repository_text() -> str:
    """Collect a bounded, lowercase signature of dependency/configuration files."""
    names = {"pyproject.toml", "requirements.txt", "package.json", "docker-compose.yml", "docker-compose.yaml"}
    parts: list[str] = []
    paths, _ = _bounded_repository_files(names=names, max_files=500,
                                         max_bytes=10_000_000, max_file_bytes=1_000_000)
    for path in paths:
        parts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(parts).lower()


@mcp.tool()
def detect_rag_stack() -> dict[str, object]:
    """Detect likely language, API, RAG, vector, and evaluation components in the selected repository."""
    text = _repository_text()
    markers = {
        "python": ["fastapi", "langchain", "llama_index", "haystack", "pytest"],
        "typescript": ["next", "express", "typescript", "langchain"],
        "orchestration": ["langchain", "llama_index", "haystack", "langgraph", "dspy"],
        "vector_store": ["qdrant", "pgvector", "chromadb", "weaviate", "milvus", "lancedb", "faiss"],
        "parsing": ["docling", "unstructured", "pymupdf", "pypdf", "tika"],
        "evaluation": ["ragas", "deepeval"],
        "observability": ["langfuse", "phoenix", "openlit"],
    }
    detected = {layer: [item for item in options if item in text] for layer, options in markers.items()}
    return {"repository": str(ROOT), "detected": detected, "is_rag_project": bool(detected["orchestration"] or detected["vector_store"])}


@mcp.tool()
def inspect_integrations() -> dict[str, object]:
    """Inventory Docker services and test setup in the selected repository without executing code."""
    return inspect_repository_integrations(ROOT)


@mcp.tool()
def inspect_docker_runtime_status(compose_file: str | None = None) -> dict[str, object]:
    """Read Docker Compose service state for the selected repository without starting or changing services."""
    return inspect_docker_runtime(ROOT, compose_file)


@mcp.tool()
def probe_selected_repository_dependencies(confirm: bool = False, compose_file: str | None = None,
                                            timeout_seconds: int = 3) -> dict[str, object]:
    """With confirm=true, probe loopback-published PostgreSQL, Qdrant, and Redis ports from Compose."""
    return probe_repository_dependencies(ROOT, compose_file, confirm, timeout_seconds)


@mcp.tool()
def run_selected_repository_tests(confirm: bool = False, timeout_seconds: int = 120) -> dict[str, object]:
    """Run detected pytest, Jest, or Vitest tests only after explicit confirmation."""
    return run_tests(ROOT, confirm, timeout_seconds)


@mcp.tool()
def rag_prepare_source_edit(relative_path: str, old_text: str, new_text: str) -> dict[str, object]:
    """Preview an exact text edit and source hash; this tool does not change files."""
    return preview_source_edit(ROOT, relative_path, old_text, new_text)


@mcp.tool()
def rag_apply_approved_source_edit(relative_path: str, expected_sha256: str, old_text: str,
                                   new_text: str, confirm: bool = False) -> dict[str, object]:
    """Apply a previously reviewed exact edit only when confirm=true and the file hash still matches."""
    return apply_approved_source_edit(ROOT, relative_path, expected_sha256, old_text, new_text, confirm)


@mcp.tool()
def recommend_rag_stack(project_size: str = "starter", storage_preference: str = "qdrant") -> dict[str, str]:
    """Recommend a compatible RAG stack using starter, postgres, or enterprise project size."""
    if project_size not in RAG_STACKS:
        raise ValueError("project_size must be starter, postgres, or enterprise")
    recommendation = RAG_STACKS[project_size].copy()
    if storage_preference.lower() in {"postgres", "pgvector"}:
        recommendation["vector_store"] = "PostgreSQL with pgvector"
    elif storage_preference.lower() == "qdrant":
        recommendation["vector_store"] = "Qdrant"
    return recommendation


@mcp.tool()
def create_rag_project_plan(project_goal: str, project_size: str = "starter") -> dict[str, object]:
    """Produce a safe, read-only implementation plan for a new or existing RAG repository."""
    stack = recommend_rag_stack(project_size)
    return {
        "goal": project_goal,
        "selected_stack": stack,
        "implementation_steps": [
            "Create document ingestion with validation, parsing, metadata, and chunking.",
            "Generate embeddings and index chunks in the selected vector store.",
            "Implement retrieval with metadata filters and an evaluation dataset.",
            "Add a FastAPI query endpoint, tests, logging, and health checks.",
            "Run Ragas or DeepEval checks in CI before deployment.",
            "Use approval-gated MCP write tools only after the plan is reviewed.",
        ],
        "safety": "This tool creates a plan only; it does not change the repository.",
    }


@mcp.tool()
def read_source_file(relative_path: str) -> str:
    """Read a source file located inside this workspace."""
    candidate = _safe_path(relative_path)
    if not candidate.is_file():
        raise ValueError("source path must be a regular file")
    if candidate.stat().st_size > MAX_SOURCE_READ_BYTES:
        raise ValueError(f"source file exceeds the {MAX_SOURCE_READ_BYTES}-byte read limit")
    return candidate.read_text(encoding="utf-8")


@mcp.tool()
def search_code(query: str, max_results: int = 25) -> list[dict[str, object]]:
    """Search workspace source files for text, returning file and line matches."""
    if not query or len(query) > 256:
        raise ValueError("query must contain 1-256 characters")
    max_results = max(1, min(max_results, 100))
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    results: list[dict[str, object]] = []
    paths, _ = _bounded_repository_files(suffixes=SAFE_SUFFIXES, max_files=MAX_SEARCH_FILES,
                                         max_bytes=MAX_SEARCH_BYTES,
                                         max_file_bytes=MAX_SOURCE_READ_BYTES)
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                results.append({"file": str(path.relative_to(ROOT)), "line": line_number, "text": line.strip()})
                if len(results) >= max_results:
                    return results
    return results


@mcp.tool()
def inspect_routes() -> list[dict[str, object]]:
    """Return FastAPI routes from this starter or decorators found in a selected Python repository."""
    if ROOT == SERVER_ROOT:
        return [{"path": route.path, "methods": sorted(route.methods or [])} for route in app.routes]
    routes: list[dict[str, object]] = []
    pattern = re.compile(r"@(?:app|router)\.(get|post|put|patch|delete)\([\"']([^\"']+)")
    paths, _ = _bounded_repository_files(suffixes={".py"}, max_files=MAX_SEARCH_FILES,
                                         max_bytes=MAX_SEARCH_BYTES,
                                         max_file_bytes=MAX_SOURCE_READ_BYTES)
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            match = pattern.search(line)
            if match:
                routes.append({"path": match.group(2), "methods": [match.group(1).upper()],
                               "file": str(path.relative_to(ROOT)), "line": line_number})
                if len(routes) >= MAX_ROUTE_RESULTS:
                    return routes
    return routes


@mcp.tool()
def inspect_dependencies() -> str:
    """Return Python dependencies declared by the project."""
    return read_source_file("pyproject.toml")


@mcp.tool()
def rag_inspect_pipeline() -> dict[str, int | str]:
    """Describe RAG storage and chunking configuration."""
    if _persistent_store is None:
        return store.metrics()
    return {"backend": "postgresql_qdrant", "collections": len(_persistent_store.collections("local-dev")),
            "embedding_model": _persistent_store.embedder.model_name,
            "chunk_size": _persistent_store.chunk_size, "chunk_overlap": _persistent_store.chunk_overlap}


@mcp.tool()
def rag_list_collections() -> dict[str, object]:
    """List RAG collections with basic statistics."""
    return {"collections": store.collections() if _persistent_store is None else _persistent_store.collections("local-dev")}


@mcp.tool()
def rag_collection_info(collection: str = "default") -> dict[str, object]:
    """Return configuration and document/chunk statistics for one collection."""
    return store.collection_info(collection) if _persistent_store is None else _persistent_store.collection_info("local-dev", collection)


@mcp.tool()
def rag_ingest_text(source: str, text: str, collection: str = "default",
                    metadata: dict[str, object] | None = None) -> dict[str, object]:
    """Ingest raw text; persistent ingestion is content-hash idempotent while memory ingestion creates each request."""
    if not text.strip():
        raise ValueError("text must not be empty")
    if len(text) > MAX_INGEST_CHARS:
        raise ValueError(f"text must not exceed {MAX_INGEST_CHARS} characters")
    if not source or len(source) > 2048:
        raise ValueError("source must contain 1-2048 characters")
    _validate_collection(collection)
    _validate_metadata(metadata)
    if _persistent_store is not None:
        return _persistent_store.ingest_text("local-dev", collection, source, text, metadata)
    return store.ingest(title=source, content=text, source=source, collection=collection, metadata=metadata)


@mcp.tool()
def rag_ingest_file(path: str, collection: str = "default", metadata: dict[str, object] | None = None) -> dict[str, object]:
    """Ingest a .txt or .md file from the explicitly selected repository only."""
    candidate = (ROOT / path).resolve()
    if ROOT not in candidate.parents or candidate.suffix.lower() not in INGEST_SUFFIXES:
        raise ValueError("path must be a workspace-relative .txt or .md file")
    if not candidate.is_file():
        raise ValueError("file does not exist")
    if candidate.stat().st_size > MAX_INGEST_CHARS * 4:
        raise ValueError("file exceeds the maximum ingest size")
    _validate_collection(collection)
    _validate_metadata(metadata)
    content = candidate.read_text(encoding="utf-8")
    if len(content) > MAX_INGEST_CHARS:
        raise ValueError(f"file text must not exceed {MAX_INGEST_CHARS} characters")
    if _persistent_store is not None:
        return _persistent_store.ingest_text("local-dev", collection, str(candidate), content, metadata)
    return store.ingest(title=candidate.name, content=content, source=str(candidate), collection=collection, metadata=metadata)


@mcp.tool()
def rag_search(query: str, collection: str = "default", top_k: int = 5,
               filters: dict[str, object] | None = None) -> dict[str, object]:
    """Search a collection with optional exact-match metadata filters."""
    _validate_query(query)
    _validate_collection(collection)
    _validate_metadata(filters)
    top_k = max(1, min(top_k, 20))
    results = (store.search(query, top_k, collection=collection, filters=filters) if _persistent_store is None
               else _persistent_store.search("local-dev", collection, query, top_k, filters))
    return {"query": query, "collection": collection, "results": results}


@mcp.tool()
def rag_delete(collection: str = "default", document_ids: list[str] | None = None,
               chunk_ids: list[str] | None = None, confirm: bool = False) -> dict[str, object]:
    """Delete documents or chunks only after explicit confirmation."""
    if not confirm:
        raise ValueError("Deletion requires confirm=True")
    _validate_collection(collection)
    ids = [*(document_ids or []), *(chunk_ids or [])]
    if len(ids) > 500:
        raise ValueError("at most 500 document and chunk IDs may be deleted in one call")
    if any(not isinstance(value, str) or not value or len(value) > 128 for value in ids):
        raise ValueError("document and chunk IDs must contain 1-128 characters")
    if not document_ids and not chunk_ids:
        raise ValueError("Supply document_ids or chunk_ids")
    if _persistent_store is not None:
        return _persistent_store.delete("local-dev", collection, document_ids or [], chunk_ids or [])
    return store.delete(collection, document_ids, chunk_ids)


@mcp.tool()
def rag_reindex_collection(collection: str = "default", embedding_model: str | None = None,
                           confirm: bool = False) -> dict[str, object]:
    """Rebuild persistent vectors with the configured embedder after explicit confirmation."""
    if not confirm:
        raise ValueError("Reindexing requires confirm=True")
    _validate_collection(collection)
    if _persistent_store is not None:
        return _persistent_store.reindex("local-dev", collection, embedding_model)
    info = store.collection_info(collection)
    if info["stats"]["documents"] == 0:
        raise ValueError("collection has no documents to reindex")
    return {"job_id": f"reindex-{collection}", "status": "not_required",
            "embedding_model": embedding_model or "not_applicable",
            "note": "The in-memory backend computes retrieval scores on demand and has no materialized index."}


@mcp.tool()
def rag_eval_query(query: str, collection: str = "default", expected_answer: str | None = None,
                   metrics: list[str] | None = None,
                   expected_document_id: str | None = None,
                   filters: dict[str, object] | None = None) -> dict[str, object]:
    """Score retrieval deterministically; persistent mode can also save dataset-run history through the API."""
    _validate_query(query)
    _validate_collection(collection)
    _validate_metadata(filters)
    if expected_answer is not None and len(expected_answer) > 10_000:
        raise ValueError("expected_answer must not exceed 10000 characters")
    if expected_document_id is not None and (not isinstance(expected_document_id, str)
                                             or len(expected_document_id) > 100):
        raise ValueError("expected_document_id must be a string of at most 100 characters")
    results = (store.search(query, collection=collection, filters=filters) if _persistent_store is None
               else _persistent_store.search("local-dev", collection, query, filters=filters))
    output_metrics = score_case(expected_answer or "", expected_document_id, results)
    supported_metrics = {"retrieval_hit", "answer_term_recall", "reciprocal_rank"}
    requested_metrics = metrics or sorted(supported_metrics)
    unsupported = set(requested_metrics) - supported_metrics
    if unsupported:
        raise ValueError(f"Unsupported metrics: {', '.join(sorted(unsupported))}")
    output_metrics = {name: value for name, value in output_metrics.items() if name in requested_metrics}
    return {"query": query, "collection": collection, "metrics": output_metrics,
            "requested_metrics": requested_metrics, "result_count": len(results)}


@mcp.tool()
def rag_eval_dataset(cases: list[dict[str, object]], collection: str = "default",
                     top_k: int = 5) -> dict[str, object]:
    """Evaluate a structured batch of query/expected_answer cases and return per-case scores and aggregates."""
    if not cases or len(cases) > 100:
        raise ValueError("cases must contain between 1 and 100 evaluation examples")
    _validate_collection(collection)
    top_k = max(1, min(top_k, 20))
    evaluated: list[dict[str, object]] = []
    for index, case in enumerate(cases):
        query_value = case.get("query", "")
        expected_value = case.get("expected_answer", "")
        expected_document_id = case.get("expected_document_id")
        metadata = case.get("metadata") or {}
        if not isinstance(query_value, str) or not isinstance(expected_value, str):
            raise ValueError(f"case {index}: query and expected_answer must be strings")
        if expected_document_id is not None and (not isinstance(expected_document_id, str)
                                                  or len(expected_document_id) > 100):
            raise ValueError(f"case {index}: expected_document_id must be a string of at most 100 characters")
        if len(expected_value) > 10_000:
            raise ValueError(f"case {index}: expected_answer must not exceed 10000 characters")
        if not isinstance(metadata, dict):
            raise ValueError(f"case {index}: metadata must be an object")
        _validate_metadata(metadata)
        query = query_value.strip()
        expected = expected_value.strip()
        try:
            _validate_query(query)
        except ValueError as error:
            raise ValueError(f"case {index}: {error}") from error
        if len(expected) > 10_000:
            raise ValueError(f"case {index}: expected_answer must not exceed 10000 characters")
        results = (store.search(query, top_k, collection=collection, filters=metadata) if _persistent_store is None
                   else _persistent_store.search("local-dev", collection, query, top_k, metadata))
        scores = score_case(expected, expected_document_id, results)
        evaluated.append({"case": index, "query": query, "result_count": len(results),
                          **scores})
    aggregate = aggregate_case_metrics(evaluated)
    return {"collection": collection, "case_count": len(evaluated), "cases": evaluated,
            "aggregate": aggregate}


@mcp.tool()
def rag_last_traces(limit: int = 20, types: list[str] | None = None) -> dict[str, object]:
    """Return recent structured MCP/RAG traces for debugging."""
    bounded_limit = max(1, min(limit, 100))
    if _persistent_store is None:
        traces = store.traces(bounded_limit, types)
    else:
        traces = _persistent_store.traces("local-dev", bounded_limit)
        if types:
            traces = [item for item in traces if item["type"].lower() in {value.lower() for value in types}]
    return {"traces": traces}


@mcp.tool()
def rag_hybrid_search(query: str, tenant_id: str, project_id: str = "default", top_k: int = 5) -> dict[str, object]:
    """Tenant-isolated dense and lexical retrieval fused with reciprocal-rank fusion."""
    if _persistent_store is None:
        raise ValueError("Set RAG_STORAGE_BACKEND=persistent and configure PostgreSQL/Qdrant to use this tool")
    _validate_query(query)
    _validate_collection(project_id)
    top_k = max(1, min(top_k, 20))
    return {"query": query, "tenant_id": tenant_id, "project_id": project_id,
            "results": _persistent_store.hybrid_search(tenant_id, project_id, query, top_k), "mode": "dense+sparse_rrf"}


@mcp.tool()
def rag_get_job(tenant_id: str, job_id: str) -> dict[str, object] | None:
    """Return a persistent ingestion job only when it belongs to the supplied tenant."""
    if _persistent_store is None:
        raise ValueError("Set RAG_STORAGE_BACKEND=persistent and configure PostgreSQL/Qdrant to use this tool")
    return _persistent_store.get_job(tenant_id, job_id)


@mcp.tool()
def rag_check_dependencies() -> dict[str, object]:
    """Check connectivity to the configured persistent storage and job-queue services."""
    if _persistent_store is None:
        return {"status": "ready", "dependencies": {"in_memory": True}}
    dependencies = _persistent_store.readiness()
    return {"status": "ready" if all(dependencies.values()) else "not_ready",
            "dependencies": dependencies}


@mcp.resource("rag://collections")
def collections_resource() -> str:
    """Read-only collection inventory resource."""
    return str(rag_list_collections())


@mcp.resource("rag://schema")
def schema_resource() -> str:
    """Read-only logical RAG data schema resource."""
    return str({"collections": ["name", "config"], "documents": ["id", "source", "metadata"],
                "chunks": ["id", "document_id", "text", "metadata"],
                "evaluation_datasets": ["id", "tenant_id", "project_id", "name", "cases"],
                "evaluation_runs": ["id", "dataset_id", "metrics", "results"],
                "traces": ["id", "tool_name", "details"]})


@mcp.prompt()
def rag_debug_search_workflow(default_collection: str = "default", default_top_k: int = 5) -> str:
    """Provide a reusable workflow for retrieval debugging."""
    return (f"Debug retrieval in collection '{default_collection}': call rag_collection_info, then rag_search "
            f"with top_k={default_top_k}, then rag_last_traces. Explain the findings before proposing a change.")


@mcp.tool()
def inspect_logs() -> dict[str, str]:
    """Describe bounded local Compose log access without returning potentially sensitive log contents."""
    return {"status": "bounded local Docker logs", "command": "docker compose logs --tail=100",
            "retention": "Each Compose service keeps at most five 10 MiB json-file log segments on the Docker host.",
            "note": "Logs can contain sensitive data. Configure an access-controlled off-host collector for centralized retention."}


if __name__ == "__main__":
    mcp.run()
