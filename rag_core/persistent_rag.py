"""End-to-end persistent RAG service: PostgreSQL metadata plus Qdrant vectors."""
from __future__ import annotations

from uuid import UUID, uuid4

from app.rag import _chunks
from rag_core.embeddings import HashEmbedder, OllamaEmbedder
from rag_core.evaluation import aggregate_case_metrics, score_case
from rag_core.storage.postgres_adapter import PostgresAdapter
from rag_core.storage.qdrant_adapter import QdrantAdapter


class PersistentRagService:
    def __init__(self, database_url: str, qdrant_url: str, collection_name: str, dimensions: int,
                 chunk_size: int, chunk_overlap: int, embedding_provider: str = "hash",
                 ollama_url: str = "http://localhost:11434", ollama_model: str = "nomic-embed-text",
                 redis_url: str | None = None, qdrant_api_key: str | None = None) -> None:
        self.postgres = PostgresAdapter(database_url)
        self.qdrant = QdrantAdapter(qdrant_url, collection_name, dimensions, qdrant_api_key)
        self.embedder = (OllamaEmbedder(ollama_url, ollama_model) if embedding_provider == "ollama"
                         else HashEmbedder(dimensions))
        self.embedding_dimensions = dimensions
        self.chunk_size, self.chunk_overlap = chunk_size, chunk_overlap
        self.redis_url = redis_url

    def ensure_storage(self) -> None:
        self.qdrant.ensure_collection()

    def readiness(self) -> dict[str, bool]:
        status = {"postgresql": False, "qdrant": False}
        try:
            status["postgresql"] = self.postgres.ping()
        except Exception:
            pass
        try:
            status["qdrant"] = self.qdrant.ping()
        except Exception:
            pass
        if self.redis_url:
            client = None
            try:
                from redis import Redis
                client = Redis.from_url(self.redis_url, socket_connect_timeout=2)
                status["redis"] = bool(client.ping())
            except Exception:
                status["redis"] = False
            finally:
                if client is not None:
                    client.close()
        return status

    def ingest_text(self, tenant_name: str, project_name: str, source: str, text: str,
                    metadata: dict[str, object] | None = None, job_id: UUID | None = None) -> dict[str, object]:
        tenant = self.postgres.get_or_create_tenant(tenant_name)
        project = self.postgres.get_or_create_project(tenant.id, project_name)
        if job_id:
            job = self.postgres.get_job(job_id)
            if job is None:
                raise ValueError("ingestion job does not exist")
            if job.tenant_id != tenant.id or job.project_id != project.id:
                raise ValueError("ingestion job does not belong to the requested tenant and project")
            self.postgres.update_job(job.id, "STARTED", progress=1)
        else:
            job = self.postgres.create_job(tenant.id, project.id, "INGEST_TEXT", {"source": source})
        try:
            self.ensure_storage()
            document, created = self.postgres.create_document_if_new(tenant.id, project.id, "text", source,
                                                                       "text/plain", text, metadata)
            created_chunks = 0
            for index, chunk_text in enumerate(_chunks(text, self.chunk_size, self.chunk_overlap)):
                chunk, is_new = self.postgres.create_chunk_if_new(tenant.id, document.id, index, chunk_text, metadata)
                self.postgres.link_document_chunk(document.id, chunk.id)
                if is_new or chunk.qdrant_point_id is None:
                    point_id = uuid4()
                    self.postgres.set_chunk_qdrant_point(chunk.id, point_id)
                else:
                    point_id = chunk.qdrant_point_id
                if not self.qdrant.point_exists(point_id):
                    point_metadata = {"source": source, **(metadata or {})}
                    vector = self.embedder.embed(chunk_text)
                    if len(vector) != self.embedding_dimensions:
                        raise ValueError(f"Embedder returned {len(vector)} dimensions; Qdrant collection expects {self.embedding_dimensions}")
                    self.qdrant.upsert_chunk(point_id, vector, tenant.id, project.id,
                                              document.id, chunk.id, index, chunk_text, point_metadata)
                    created_chunks += 1
            status = "completed" if created or created_chunks else "duplicate"
            result = {"status": status, "document_id": str(document.id), "chunks_created": created_chunks}
            self.postgres.update_job(job.id, "SUCCESS", result)
            self.postgres.write_trace(tenant.id, project.id, "INGEST", "rag_ingest_text", {"source": source}, result)
            return {"job_id": str(job.id), **result}
        except Exception as error:
            self.postgres.update_job(job.id, "FAILED", error=str(error), progress=0)
            raise

    def create_ingest_job(self, tenant_name: str, project_name: str, source: str) -> dict[str, str]:
        tenant = self.postgres.get_or_create_tenant(tenant_name)
        project = self.postgres.get_or_create_project(tenant.id, project_name)
        job = self.postgres.create_job(tenant.id, project.id, "INGEST_TEXT",
                                       {"source": source})
        return {"job_id": str(job.id), "status": job.status}

    def search(self, tenant_name: str, project_name: str, query: str, top_k: int = 5,
               filters: dict[str, object] | None = None) -> list[dict[str, object]]:
        tenant = self.postgres.get_or_create_tenant(tenant_name)
        project = self.postgres.get_or_create_project(tenant.id, project_name)
        self.ensure_storage()
        vector = self.embedder.embed(query)
        if len(vector) != self.embedding_dimensions:
            raise ValueError(f"Embedder returned {len(vector)} dimensions; Qdrant collection expects {self.embedding_dimensions}")
        results = self.qdrant.search_chunks(tenant.id, vector, top_k, project.id, filters)
        output = [{"document_id": item["payload"]["document_id"], "chunk_id": item["payload"]["chunk_id"],
                   "title": item["payload"].get("source", "source"), "content": item["payload"]["text"],
                   "text": item["payload"]["text"], "score": item["score"],
                   "metadata": item["payload"].get("metadata", {})} for item in results]
        self.postgres.write_trace(tenant.id, project.id, "SEARCH", "rag_search",
                                  {"query_length": len(query)}, {"count": len(output)})
        return output

    def hybrid_search(self, tenant_name: str, project_name: str, query: str, top_k: int = 5) -> list[dict[str, object]]:
        tenant = self.postgres.get_or_create_tenant(tenant_name)
        project = self.postgres.get_or_create_project(tenant.id, project_name)
        dense = self.search(tenant_name, project_name, query, max(top_k * 3, top_k))
        sparse = self.postgres.lexical_search(tenant.id, project.id, query, max(top_k * 3, top_k))
        fused: dict[str, tuple[dict[str, object], float]] = {}
        for ranking in (dense, sparse):
            for rank, item in enumerate(ranking, 1):
                key = str(item["chunk_id"])
                prior, score = fused.get(key, (item, 0.0))
                fused[key] = (prior, score + 1 / (60 + rank))
        output = [{**item, "score": round(score, 6)} for item, score in
                  sorted(fused.values(), key=lambda pair: pair[1], reverse=True)[:top_k]]
        self.postgres.write_trace(tenant.id, project.id, "SEARCH", "rag_hybrid_search",
                                  {"query_length": len(query)}, {"count": len(output), "dense_count": len(dense),
                                                      "sparse_count": len(sparse)})
        return output

    def get_job(self, tenant_name: str, job_id: str) -> dict[str, object] | None:
        try:
            parsed_job_id = UUID(job_id)
        except ValueError:
            return None
        tenant = self.postgres.get_tenant(tenant_name)
        job = None if tenant is None else self.postgres.get_job_for_tenant(parsed_job_id, tenant.id)
        if job is None:
            return None
        return {"id": str(job.id), "status": job.status, "type": job.type, "progress": job.progress,
                "outputs": job.outputs, "error": job.error}

    def list_documents(self, tenant_name: str, project_name: str,
                       limit: int = 100, offset: int = 0) -> list[dict[str, object]]:
        tenant = self.postgres.get_tenant(tenant_name)
        if tenant is None:
            return []
        project = self.postgres.get_project(tenant.id, project_name)
        if project is None:
            return []
        return self.postgres.list_documents(tenant.id, project.id, limit, offset)

    def collections(self, tenant_name: str) -> list[dict[str, object]]:
        tenant = self.postgres.get_tenant(tenant_name)
        return [] if tenant is None else self.postgres.list_projects(tenant.id)

    def collection_info(self, tenant_name: str, project_name: str) -> dict[str, object]:
        collections = self.collections(tenant_name)
        for item in collections:
            if item["name"] == project_name:
                return {"collection": project_name,
                        "config": {"storage": "postgresql_qdrant", "embedding_model": self.embedder.model_name},
                        "stats": {"documents": item["documents"], "points": item["chunks"]}}
        return {"collection": project_name,
                "config": {"storage": "postgresql_qdrant", "embedding_model": self.embedder.model_name},
                "stats": {"documents": 0, "points": 0}}

    def delete(self, tenant_name: str, project_name: str, document_ids: list[str],
               chunk_ids: list[str]) -> dict[str, int]:
        """Delete only IDs owned by the named tenant/project, removing vectors before metadata."""
        tenant = self.postgres.get_tenant(tenant_name)
        project = None if tenant is None else self.postgres.get_project(tenant.id, project_name)
        if tenant is None or project is None:
            return {"documents_deleted": 0, "chunks_deleted": 0}
        try:
            parsed_documents = [UUID(value) for value in document_ids]
            parsed_chunks = [UUID(value) for value in chunk_ids]
        except (ValueError, TypeError, AttributeError) as error:
            raise ValueError("persistent document and chunk IDs must be UUIDs") from error
        targets = self.postgres.resolve_delete_targets(tenant.id, project.id, parsed_documents, parsed_chunks)
        self.qdrant.delete_points(targets["point_ids"])
        result = self.postgres.delete_resolved_targets(tenant.id, project.id, targets)
        self.postgres.write_trace(tenant.id, project.id, "DELETE", "rag_delete",
                                  {"document_count": len(parsed_documents), "chunk_count": len(parsed_chunks)}, result)
        return result

    def reindex(self, tenant_name: str, project_name: str,
                embedding_model: str | None = None) -> dict[str, object]:
        """Recompute and upsert every project vector using the configured embedder."""
        if embedding_model and embedding_model != self.embedder.model_name:
            raise ValueError(f"requested embedding model {embedding_model!r} does not match configured model "
                             f"{self.embedder.model_name!r}; change service configuration before reindexing")
        tenant = self.postgres.get_tenant(tenant_name)
        project = None if tenant is None else self.postgres.get_project(tenant.id, project_name)
        if tenant is None or project is None:
            raise ValueError("collection has no documents to reindex")
        chunks = self.postgres.list_chunks_for_project(tenant.id, project.id)
        if not chunks:
            raise ValueError("collection has no documents to reindex")
        self.ensure_storage()
        for item in chunks:
            point_id = item["point_id"]
            if point_id is None:
                point_id = uuid4()
                self.postgres.set_chunk_qdrant_point(item["chunk_id"], point_id)
            vector = self.embedder.embed(item["text"])
            if len(vector) != self.embedding_dimensions:
                raise ValueError(f"Embedder returned {len(vector)} dimensions; Qdrant collection expects "
                                 f"{self.embedding_dimensions}")
            metadata = {"source": item["source"], **(item["metadata"] or {})}
            self.qdrant.upsert_chunk(point_id, vector, tenant.id, project.id,
                                     item["document_id"], item["chunk_id"], item["chunk_index"],
                                     item["text"], metadata)
        result = {"status": "completed", "chunks_reindexed": len(chunks),
                  "embedding_model": self.embedder.model_name}
        self.postgres.write_trace(tenant.id, project.id, "REINDEX", "rag_reindex_collection",
                                  {"chunk_count": len(chunks), "embedding_model": self.embedder.model_name}, result)
        return result

    def traces(self, tenant_name: str, limit: int = 20) -> list[dict[str, object]]:
        tenant = self.postgres.get_tenant(tenant_name)
        return [] if tenant is None else self.postgres.list_traces(tenant.id, limit)

    def create_evaluation_dataset(self, tenant_name: str, project_name: str, name: str,
                                  description: str, cases: list[dict[str, object]]) -> dict[str, object]:
        tenant = self.postgres.get_or_create_tenant(tenant_name)
        project = self.postgres.get_or_create_project(tenant.id, project_name)
        return self.postgres.create_evaluation_dataset(tenant.id, project.id, name, description, cases)

    def list_evaluation_datasets(self, tenant_name: str, project_name: str,
                                 limit: int = 100, offset: int = 0) -> list[dict[str, object]]:
        tenant = self.postgres.get_tenant(tenant_name)
        if tenant is None:
            return []
        project = self.postgres.get_project(tenant.id, project_name)
        if project is None:
            return []
        return self.postgres.list_evaluation_datasets(tenant.id, project.id, limit, offset)

    def get_evaluation_dataset(self, tenant_name: str, project_name: str,
                               dataset_id: str) -> dict[str, object] | None:
        try:
            parsed_id = UUID(dataset_id)
        except ValueError:
            return None
        tenant = self.postgres.get_tenant(tenant_name)
        if tenant is None:
            return None
        project = self.postgres.get_project(tenant.id, project_name)
        if project is None:
            return None
        return self.postgres.get_evaluation_dataset(tenant.id, project.id, parsed_id)

    def run_evaluation_dataset(self, tenant_name: str, project_name: str, dataset_id: str,
                               top_k: int = 5) -> dict[str, object] | None:
        dataset = self.get_evaluation_dataset(tenant_name, project_name, dataset_id)
        if dataset is None:
            return None
        tenant = self.postgres.get_tenant(tenant_name)
        assert tenant is not None
        project = self.postgres.get_or_create_project(tenant.id, project_name)
        case_results: list[dict[str, object]] = []
        for case in dataset["cases"]:
            case_filters = case.get("metadata") or {}
            results = self.search(tenant_name, project_name, str(case["query"]), top_k,
                                  case_filters if isinstance(case_filters, dict) else None)
            scores = score_case(str(case["expected_answer"]), case.get("expected_document_id"), results)
            case_results.append({"case_id": case["id"], "query": case["query"],
                                 **scores,
                                 "retrieved": [{"document_id": item["document_id"], "chunk_id": item["chunk_id"],
                                                "score": item["score"]} for item in results]})
        metrics = aggregate_case_metrics(case_results)
        run = self.postgres.create_evaluation_run(tenant.id, project.id, UUID(dataset_id), metrics, case_results)
        self.postgres.write_trace(tenant.id, project.id, "EVALUATION", "rag_eval_dataset",
                                  {"dataset_id": dataset_id}, metrics)
        return run

    def list_evaluation_runs(self, tenant_name: str, project_name: str,
                             dataset_id: str, limit: int = 100, offset: int = 0) -> list[dict[str, object]]:
        dataset = self.get_evaluation_dataset(tenant_name, project_name, dataset_id)
        if dataset is None:
            return []
        tenant = self.postgres.get_tenant(tenant_name)
        assert tenant is not None
        project = self.postgres.get_project(tenant.id, project_name)
        if project is None:
            return []
        return self.postgres.list_evaluation_runs(tenant.id, project.id, UUID(dataset_id), limit, offset)
