from types import SimpleNamespace

import pytest
import httpx
from qdrant_client import models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

import rag_core.storage.qdrant_adapter as qdrant_module
from rag_core.storage.qdrant_adapter import QdrantAdapter
from rag_core.errors import TransientDependencyError


class FakeQdrantClient:
    def __init__(self, size: int, payload_schema: dict[str, object]) -> None:
        self.info = SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(
                vectors=models.VectorParams(size=size, distance=models.Distance.COSINE))),
            payload_schema=payload_schema,
        )
        self.created_indexes: list[str] = []

    def collection_exists(self, name: str) -> bool:
        return True

    def get_collection(self, name: str):
        return self.info

    def create_payload_index(self, collection, field, schema, wait=True) -> None:
        self.created_indexes.append(field)


def _adapter(size: int, schema: dict[str, object]) -> QdrantAdapter:
    adapter = QdrantAdapter.__new__(QdrantAdapter)
    adapter.client = FakeQdrantClient(size, schema)
    adapter.collection_name = "chunks"
    adapter.vector_size = 8
    return adapter


def test_ensure_collection_adds_only_missing_payload_indexes() -> None:
    adapter = _adapter(8, {"tenant_id": object()})

    adapter.ensure_collection()

    assert adapter.client.created_indexes == ["project_id"]


def test_ensure_collection_rejects_embedding_dimension_mismatch() -> None:
    adapter = _adapter(768, {})

    with pytest.raises(ValueError, match="vector size is 768"):
        adapter.ensure_collection()

    assert adapter.client.created_indexes == []


def test_qdrant_adapter_passes_optional_api_key_to_client(monkeypatch) -> None:
    captured = {}

    def fake_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(qdrant_module, "QdrantClient", fake_client)
    QdrantAdapter("https://vectors.example", "chunks", 8, "secret-key")

    assert captured == {"url": "https://vectors.example", "api_key": "secret-key"}


@pytest.mark.parametrize("failure", [
    ResponseHandlingException(httpx.ConnectError("unreachable")),
    UnexpectedResponse(503, "Unavailable", b"temporarily unavailable", httpx.Headers()),
    UnexpectedResponse(429, "Too Many Requests", b"retry later", httpx.Headers()),
])
def test_qdrant_transient_transport_and_server_errors_are_retryable(failure) -> None:
    adapter = QdrantAdapter.__new__(QdrantAdapter)
    adapter.client = SimpleNamespace(request=lambda: (_ for _ in ()).throw(failure))

    with pytest.raises(TransientDependencyError):
        adapter._call("request")


def test_qdrant_client_errors_are_not_classified_as_transient() -> None:
    failure = UnexpectedResponse(400, "Bad Request", b"invalid request", httpx.Headers())
    adapter = QdrantAdapter.__new__(QdrantAdapter)
    adapter.client = SimpleNamespace(request=lambda: (_ for _ in ()).throw(failure))

    with pytest.raises(UnexpectedResponse):
        adapter._call("request")
