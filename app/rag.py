"""Collection-aware RAG core with a local development storage adapter."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import sqrt
from re import findall
from uuid import uuid4


def _tokens(text: str) -> list[str]:
    return findall(r"[a-zA-Z0-9_]+", text.lower())


def _chunks(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    return [text[start : start + size] for start in range(0, len(text), max(1, size - overlap))]


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    magnitude = sqrt(sum(value * value for value in left.values())) * sqrt(sum(value * value for value in right.values()))
    return dot / magnitude if magnitude else 0.0


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class Chunk:
    collection: str
    document_id: str
    title: str
    chunk_id: str
    content: str
    metadata: dict[str, object] = field(default_factory=dict)


class InMemoryRagStore:
    """Local adapter whose API can later be replaced by Qdrant/PostgreSQL adapters."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80) -> None:
        self.chunk_size, self.chunk_overlap = chunk_size, chunk_overlap
        self._documents: dict[str, dict[str, dict[str, object]]] = {}
        self._chunks: list[Chunk] = []
        self._traces: list[dict[str, object]] = []

    def _trace(self, tool_name: str, details: dict[str, object]) -> None:
        self._traces.append({"id": str(uuid4()), "type": tool_name.removeprefix("rag_").split("_")[0],
                             "tool_name": tool_name, "timestamp": _now(), "details": details})
        self._traces = self._traces[-200:]

    def ingest(self, title: str, content: str, source: str | None = None, collection: str = "default",
               metadata: dict[str, object] | None = None) -> dict[str, object]:
        metadata = metadata or {}
        document_id, pieces = str(uuid4()), _chunks(content, self.chunk_size, self.chunk_overlap)
        self._documents.setdefault(collection, {})[document_id] = {"title": title, "source": source,
            "content": content, "metadata": metadata, "created_at": _now()}
        self._chunks.extend(Chunk(collection, document_id, title, f"{document_id}:{index}", piece, metadata)
                            for index, piece in enumerate(pieces))
        result = {"job_id": str(uuid4()), "status": "completed", "document_id": document_id,
                  "documents_created": 1, "chunks_created": len(pieces), "collection": collection}
        self._trace("rag_ingest_text", result)
        return result

    def search(self, query: str, limit: int = 5, collection: str = "default",
               filters: dict[str, object] | None = None) -> list[dict[str, object]]:
        filters, query_vector = filters or {}, Counter(_tokens(query))
        candidates = [chunk for chunk in self._chunks if chunk.collection == collection and
                      all(chunk.metadata.get(key) == value for key, value in filters.items())]
        ranked = sorted(((chunk, _cosine(query_vector, Counter(_tokens(chunk.content)))) for chunk in candidates),
                        key=lambda item: item[1], reverse=True)
        results = [{"document_id": chunk.document_id, "title": chunk.title, "chunk_id": chunk.chunk_id,
                    "content": chunk.content, "text": chunk.content, "score": round(score, 4),
                    "metadata": chunk.metadata} for chunk, score in ranked[:limit] if score > 0]
        self._trace("rag_search", {"query_length": len(query), "collection": collection,
                                    "top_k": limit, "result_count": len(results)})
        return results

    def documents(self, collection: str = "default", limit: int = 100, offset: int = 0) -> list[dict[str, object]]:
        records = [{"document_id": key, **value} for key, value in self._documents.get(collection, {}).items()]
        records.sort(key=lambda item: (str(item.get("created_at", "")), str(item["document_id"])), reverse=True)
        return records[offset:offset + limit]

    def collections(self) -> list[dict[str, object]]:
        names = sorted(set(self._documents) | {chunk.collection for chunk in self._chunks})
        return [{"name": name, "point_count": sum(chunk.collection == name for chunk in self._chunks),
                 "config": {"chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap,
                            "storage": "in_memory"}} for name in names]

    def collection_info(self, collection: str) -> dict[str, object]:
        return {"collection": collection, "config": {"chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap,
                "storage": "in_memory"}, "stats": {"documents": len(self._documents.get(collection, {})),
                "points": sum(chunk.collection == collection for chunk in self._chunks)}}

    def delete(self, collection: str, document_ids: list[str] | None = None,
               chunk_ids: list[str] | None = None) -> dict[str, int]:
        document_ids, chunk_ids = set(document_ids or []), set(chunk_ids or [])
        affected = {chunk.document_id for chunk in self._chunks if chunk.collection == collection and
                    (chunk.document_id in document_ids or chunk.chunk_id in chunk_ids)}
        before = len(self._chunks)
        self._chunks = [chunk for chunk in self._chunks if not (chunk.collection == collection and
                        (chunk.document_id in document_ids or chunk.chunk_id in chunk_ids))]
        for document_id in affected:
            if not any(chunk.document_id == document_id for chunk in self._chunks):
                self._documents.get(collection, {}).pop(document_id, None)
        result = {"deleted_documents": len(affected), "deleted_chunks": before - len(self._chunks)}
        self._trace("rag_delete", {"collection": collection, **result})
        return result

    def traces(self, limit: int = 20, types: list[str] | None = None) -> list[dict[str, object]]:
        items = self._traces if not types else [trace for trace in self._traces if trace["type"] in types]
        return list(reversed(items[-limit:]))

    def metrics(self) -> dict[str, int | str]:
        return {"backend": "in_memory", "collections": len(self.collections()),
                "documents": sum(len(items) for items in self._documents.values()), "chunks": len(self._chunks),
                "chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap}
