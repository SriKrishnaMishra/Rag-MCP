import asyncio
from fastapi.testclient import TestClient
import pytest
from starlette.requests import Request

from app import main
from app.main import app, store
from app.schemas import DocumentCreate, SearchRequest
from pydantic import ValidationError


client = TestClient(app)


def setup_function() -> None:
    store._documents.clear()
    store._chunks.clear()
    store._traces.clear()


def test_health_reports_rag_status() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.headers["X-Request-ID"]


def test_http_logs_correlate_without_recording_request_details(caplog) -> None:
    with caplog.at_level("INFO", logger="app.main"):
        response = client.get("/health?private_query=do-not-log")
    assert response.headers["X-Request-ID"] in caplog.text
    assert "private_query" not in caplog.text
    assert "do-not-log" not in caplog.text


def test_http_failure_logs_exception_type_without_exception_text(caplog) -> None:
    request = Request({"type": "http", "method": "GET", "path": "/private",
                       "headers": [], "query_string": b""})

    async def call_next(_request):
        raise RuntimeError("password=do-not-log-this")

    with caplog.at_level("ERROR", logger="app.main"), pytest.raises(RuntimeError):
        asyncio.run(main.request_telemetry(request, call_next))

    assert "RuntimeError" in caplog.text
    assert "do-not-log-this" not in caplog.text
    assert "password=" not in caplog.text


def test_direct_ingestion_and_search_payloads_have_size_bounds() -> None:
    with pytest.raises(ValidationError):
        DocumentCreate(title="large", content="x" * 5_000_001)
    with pytest.raises(ValidationError):
        SearchRequest(query="x" * 4001)
    with pytest.raises(ValidationError):
        DocumentCreate(title="metadata", content="body", metadata={"x" * 129: "value"})
    with pytest.raises(ValidationError):
        SearchRequest(query="bounded", filters={f"key-{index}": "value" for index in range(33)})


def test_ingest_then_search() -> None:
    created = client.post("/documents", json={
        "title": "Qdrant guide", "content": "Qdrant is a vector database for similarity search.",
        "metadata": {"team": "search"},
    })
    assert created.status_code == 201
    results = client.post("/search", json={"query": "vector similarity database"})
    assert results.status_code == 200
    assert results.json()[0]["title"] == "Qdrant guide"
    assert results.json()[0]["metadata"] == {"team": "search"}


def test_document_listing_is_paginated_newest_first() -> None:
    for title in ("oldest", "middle", "newest"):
        response = client.post("/documents", json={"title": title, "content": f"content for {title}"})
        assert response.status_code == 201

    page = client.get("/documents?limit=1&offset=1")

    assert page.status_code == 200
    assert len(page.json()) == 1
    assert page.json()[0]["title"] == "middle"
    assert client.get("/documents?offset=1000001").status_code == 422


def test_persistent_ask_forwards_metadata_filters(monkeypatch) -> None:
    class FakePersistentStore:
        received_filters = None

        def search(self, tenant, collection, query, limit, filters):
            self.received_filters = filters
            return []

    fake_store = FakePersistentStore()
    monkeypatch.setattr(main, "_persistent_store", fake_store)
    monkeypatch.setattr(main.ollama, "answer", lambda query, sources, model: "No supporting source.")

    response = client.post("/ask", json={"query": "vector", "filters": {"team": "search"}})

    assert response.status_code == 200
    assert fake_store.received_filters == {"team": "search"}


def test_evaluation_list_endpoints_forward_bounded_pagination(monkeypatch) -> None:
    class FakePersistentStore:
        dataset_page = None
        run_page = None

        def list_evaluation_datasets(self, tenant, collection, limit, offset):
            self.dataset_page = (limit, offset)
            return []

        def get_evaluation_dataset(self, tenant, collection, dataset_id):
            return {"id": dataset_id}

        def list_evaluation_runs(self, tenant, collection, dataset_id, limit, offset):
            self.run_page = (limit, offset)
            return []

    fake_store = FakePersistentStore()
    monkeypatch.setattr(main, "_persistent_store", fake_store)

    datasets = client.get("/evaluation/datasets?limit=20&offset=40")
    runs = client.get("/evaluation/datasets/test-dataset/runs?limit=10&offset=5")

    assert datasets.status_code == 200
    assert runs.status_code == 200
    assert fake_store.dataset_page == (20, 40)
    assert fake_store.run_page == (10, 5)
