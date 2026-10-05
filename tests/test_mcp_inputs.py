import pytest
from uuid import uuid4

from mcp_servers.inspector import server


@pytest.mark.parametrize("collection", ["", "../other", "x" * 65, "has space"])
def test_mcp_rejects_invalid_collection_names(collection):
    with pytest.raises(ValueError, match="collection"):
        server._validate_collection(collection)


def test_mcp_caps_query_length():
    with pytest.raises(ValueError, match="4000"):
        server._validate_query("q" * 4001)


@pytest.mark.parametrize("metadata", [{"x": 1}, {"x" * 129: "value"}, {"x": "v" * 513},
                                       {f"key-{index}": "value" for index in range(33)}])
def test_mcp_validates_metadata_shape(metadata):
    with pytest.raises(ValueError, match="metadata"):
        server._validate_metadata(metadata)


def test_mcp_text_ingestion_rejects_oversized_input_before_storage():
    with pytest.raises(ValueError, match="5000000"):
        server.rag_ingest_text("source", "x" * 5_000_001)


def test_mcp_delete_caps_id_batch_before_mutation():
    with pytest.raises(ValueError, match="at most 500"):
        server.rag_delete(document_ids=["doc"] * 501, confirm=True)


def test_mcp_evaluation_uses_metadata_filters_and_expected_document(monkeypatch):
    monkeypatch.setattr(server, "_persistent_store", None)
    collection = f"eval-{uuid4().hex[:16]}"
    document = server.store.ingest("source", "blue deployment manual", collection=collection,
                                   metadata={"team": "platform"})

    result = server.rag_eval_dataset([{"query": "blue deployment", "expected_answer": "manual",
                                      "expected_document_id": document["document_id"],
                                      "metadata": {"team": "platform"}}], collection)
    filtered_out = server.rag_eval_dataset([{"query": "blue deployment", "expected_answer": "manual",
                                             "expected_document_id": document["document_id"],
                                             "metadata": {"team": "other"}}], collection)

    assert result["cases"][0]["reciprocal_rank"] == 1.0
    assert filtered_out["cases"][0]["retrieval_hit"] == 0.0
    assert filtered_out["cases"][0]["reciprocal_rank"] == 0.0


def test_mcp_single_query_evaluation_accepts_expected_document_and_filters(monkeypatch):
    monkeypatch.setattr(server, "_persistent_store", None)
    collection = f"eval-{uuid4().hex[:16]}"
    document = server.store.ingest("source", "red network guide", collection=collection,
                                   metadata={"team": "network"})

    matching = server.rag_eval_query("red network", collection, "guide", ["reciprocal_rank"],
                                     document["document_id"], {"team": "network"})
    mismatch = server.rag_eval_query("red network", collection, "guide", ["reciprocal_rank"],
                                     document["document_id"], {"team": "security"})

    assert matching["metrics"]["reciprocal_rank"] == 1.0
    assert mismatch["metrics"]["reciprocal_rank"] == 0.0
    assert mismatch["result_count"] == 0


def test_mcp_file_ingestion_checks_size_before_reading(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    oversized = root / "large.txt"
    oversized.write_bytes(b"x" * (server.MAX_INGEST_CHARS * 4 + 1))
    monkeypatch.setattr(server, "ROOT", root)

    with pytest.raises(ValueError, match="maximum ingest size"):
        server.rag_ingest_file("large.txt")


def test_inspector_read_and_search_enforce_source_bounds(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "large.py").write_bytes(b"x" * (server.MAX_SOURCE_READ_BYTES + 1))
    (root / "small.py").write_text("needle\n@app.get('/health')\ndef health(): pass\n", encoding="utf-8")
    monkeypatch.setattr(server, "ROOT", root)

    with pytest.raises(ValueError, match="read limit"):
        server.read_source_file("large.py")
    assert server.search_code("needle") == [{"file": "small.py", "line": 1, "text": "needle"}]
    assert server.inspect_routes() == [{"path": "/health", "methods": ["GET"], "file": "small.py", "line": 2}]
    with pytest.raises(ValueError, match="1-256"):
        server.search_code("")


def test_project_inventory_caps_file_traversal_and_reports_truncation(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("a", encoding="utf-8")
    (root / "b.py").write_text("b", encoding="utf-8")
    ignored = root / "node_modules"
    ignored.mkdir()
    (ignored / "ignored.js").write_text("secret", encoding="utf-8")
    monkeypatch.setattr(server, "ROOT", root)
    monkeypatch.setattr(server, "MAX_PROJECT_FILES", 1)

    result = server.inspect_project()

    assert result["file_count"] == 1
    assert result["scan_truncated"] is True
    assert result["files"] == ["a.py"]
