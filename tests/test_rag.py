from app.rag import InMemoryRagStore


def test_store_reports_chunk_metrics() -> None:
    store = InMemoryRagStore(chunk_size=10, chunk_overlap=2)
    result = store.ingest("sample", "one two three four five six")
    assert result["chunks_created"] > 1
    assert store.metrics()["documents"] == 1


def test_collection_filter_and_delete_core() -> None:
    store = InMemoryRagStore()
    result = store.ingest("guide", "Qdrant stores vectors", collection="docs", metadata={"team": "ai"})
    assert len(store.search("vectors", collection="docs", filters={"team": "ai"})) == 1
    assert store.delete("docs", document_ids=[result["document_id"]])["deleted_documents"] == 1


def test_search_trace_records_query_length_without_raw_query_text() -> None:
    store = InMemoryRagStore()
    query = "private customer phrase"

    store.search(query)
    trace = store.traces()[0]

    assert trace["details"]["query_length"] == len(query)
    assert "query" not in trace["details"]
    assert query not in str(trace)
