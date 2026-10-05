"""Qdrant vector storage with mandatory tenant payload filtering."""
from __future__ import annotations

from uuid import UUID

import httpx
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from rag_core.errors import TransientDependencyError


class QdrantAdapter:
    def __init__(self, url: str, collection_name: str = "rag_chunks", vector_size: int = 384,
                 api_key: str | None = None) -> None:
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name, self.vector_size = collection_name, vector_size

    def _call(self, method: str, *args, **kwargs):
        try:
            return getattr(self.client, method)(*args, **kwargs)
        except ResponseHandlingException as error:
            if isinstance(error.source, (httpx.TransportError, TimeoutError, ConnectionError)):
                raise TransientDependencyError(
                    "Qdrant is temporarily unreachable; retry the request.") from error
            raise
        except UnexpectedResponse as error:
            if error.status_code == 429 or (error.status_code is not None and error.status_code >= 500):
                raise TransientDependencyError(
                    "Qdrant returned a temporary server error; retry the request.") from error
            raise

    def ensure_collection(self) -> None:
        created = not self._call("collection_exists", self.collection_name)
        if created:
            self._call("create_collection", self.collection_name, vectors_config=models.VectorParams(
                size=self.vector_size, distance=models.Distance.COSINE))
        info = self._call("get_collection", self.collection_name)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            raise ValueError("The configured Qdrant collection must use one unnamed dense vector")
        actual_size = getattr(vectors, "size", None)
        if actual_size != self.vector_size:
            raise ValueError(f"Qdrant collection vector size is {actual_size}; configured embedder size is {self.vector_size}")
        payload_schema = getattr(info, "payload_schema", None) or {}
        for field_name in ("tenant_id", "project_id"):
            if created or field_name not in payload_schema:
                self._call("create_payload_index", self.collection_name, field_name,
                           models.PayloadSchemaType.KEYWORD, wait=True)

    def ping(self) -> bool:
        self._call("get_collections")
        return True

    def point_exists(self, point_id: UUID) -> bool:
        return bool(self._call("retrieve", self.collection_name, ids=[str(point_id)], with_payload=False,
                               with_vectors=False))

    def delete_points(self, point_ids: list[UUID]) -> None:
        """Remove known point IDs; callers must resolve IDs within their tenant/project scope first."""
        if point_ids:
            self._call("delete", self.collection_name,
                       points_selector=models.PointIdsList(points=[str(point_id) for point_id in point_ids]),
                       wait=True)

    def upsert_chunk(self, point_id: UUID, vector: list[float], tenant_id: UUID, project_id: UUID,
                     document_id: UUID, chunk_id: UUID, chunk_index: int, text: str,
                     metadata: dict[str, object] | None = None) -> None:
        metadata = metadata or {}
        payload = {"tenant_id": str(tenant_id), "project_id": str(project_id), "document_id": str(document_id),
                   "chunk_id": str(chunk_id), "chunk_index": chunk_index, "text": text,
                   "source": metadata.get("source", "source"), "metadata": metadata}
        self._call("upsert", self.collection_name,
                   [models.PointStruct(id=str(point_id), vector=vector, payload=payload)], wait=True)

    def search_chunks(self, tenant_id: UUID, query_vector: list[float], top_k: int = 5,
                      project_id: UUID | None = None,
                      filters: dict[str, object] | None = None) -> list[dict[str, object]]:
        conditions = [models.FieldCondition(key="tenant_id", match=models.MatchValue(value=str(tenant_id)))]
        if project_id:
            conditions.append(models.FieldCondition(key="project_id", match=models.MatchValue(value=str(project_id))))
        for key, value in (filters or {}).items():
            if key in {"tenant_id", "project_id", "document_id", "chunk_id", "text", "metadata"}:
                continue
            conditions.append(models.FieldCondition(key=f"metadata.{key}", match=models.MatchValue(value=value)))
        response = self._call("query_points", self.collection_name, query=query_vector,
                              query_filter=models.Filter(must=conditions), limit=top_k,
                              with_payload=True)
        return [{"point_id": str(point.id), "score": point.score, "payload": point.payload} for point in response.points]
