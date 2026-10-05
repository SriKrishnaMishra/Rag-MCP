from types import SimpleNamespace
from uuid import uuid4

import pytest

from rag_core.persistent_rag import PersistentRagService
from rag_core.embeddings import HashEmbedder
from sqlalchemy.exc import OperationalError
import httpx
from workers.tasks import ingest_text
from workers.celery_app import celery_app
from rag_core.errors import TransientDependencyError
from rag_core.storage.postgres_adapter import PostgresAdapter


def test_ingestion_task_retries_transient_storage_and_transport_errors() -> None:
    assert set(ingest_text.autoretry_for) == {
        ConnectionError, TimeoutError, OperationalError, httpx.TransportError, TransientDependencyError,
    }
    assert ingest_text.max_retries == 3


def test_celery_redelivers_worker_lost_ingestion_with_bounded_prefetch() -> None:
    assert celery_app.conf.task_acks_late is True
    assert celery_app.conf.task_reject_on_worker_lost is True
    assert celery_app.conf.worker_prefetch_multiplier == 1
    assert celery_app.conf.broker_transport_options["visibility_timeout"] == 21600
    assert celery_app.conf.result_backend_transport_options["visibility_timeout"] == 21600
    assert celery_app.conf.visibility_timeout == 21600
    assert celery_app.conf.accept_content == ["json"]
    assert celery_app.conf.task_serializer == celery_app.conf.result_serializer == "json"


def test_persistent_read_lists_do_not_create_tenants_or_projects() -> None:
    class ReadOnlyPostgres:
        def get_tenant(self, _name):
            return None
        def get_or_create_tenant(self, _name):
            raise AssertionError("a read-only list must not create a tenant")
        def get_or_create_project(self, _tenant_id, _name):
            raise AssertionError("a read-only list must not create a project")

    service = PersistentRagService.__new__(PersistentRagService)
    service.postgres = ReadOnlyPostgres()

    assert service.list_documents("missing-tenant", "missing-project") == []
    assert service.list_evaluation_datasets("missing-tenant", "missing-project") == []
    assert service.get_evaluation_dataset("missing-tenant", "missing-project", str(uuid4())) is None


def test_persistent_search_traces_store_query_length_not_query_text() -> None:
    class TracePostgres:
        tenant = SimpleNamespace(id=uuid4())
        project = SimpleNamespace(id=uuid4())
        def get_or_create_tenant(self, _name): return self.tenant
        def get_or_create_project(self, _tenant_id, _name): return self.project
        def write_trace(self, *args, **kwargs): self.trace = args
        def lexical_search(self, *_args): return []

    class EmptyQdrant:
        def search_chunks(self, *_args): return []

    class FixedEmbedder:
        def embed(self, _query): return [0.0] * 8

    service = PersistentRagService.__new__(PersistentRagService)
    service.postgres = TracePostgres()
    service.qdrant = EmptyQdrant()
    service.embedder = FixedEmbedder()
    service.embedding_dimensions = 8
    service.ensure_storage = lambda: None
    query = "private customer phrase"

    service.hybrid_search("tenant", "project", query)

    trace_inputs = service.postgres.trace[4]
    assert trace_inputs == {"query_length": len(query)}
    assert query not in str(service.postgres.trace)


def test_legacy_trace_query_text_is_redacted_when_read() -> None:
    query = "legacy private phrase"

    sanitized = PostgresAdapter._sanitize_trace_inputs({"query": query, "collection": "docs"})

    assert sanitized == {"collection": "docs", "query_length": len(query)}
    assert query not in str(sanitized)


def test_persistent_read_lists_do_not_create_tenants_or_projects() -> None:
    class ReadOnlyPostgres:
        def get_tenant(self, _name):
            return None
        def get_or_create_tenant(self, _name):
            raise AssertionError("a read-only list must not create a tenant")
        def get_or_create_project(self, _tenant_id, _name):
            raise AssertionError("a read-only list must not create a project")

    service = PersistentRagService.__new__(PersistentRagService)
    service.postgres = ReadOnlyPostgres()

    assert service.list_documents("missing-tenant", "missing-project") == []
    assert service.list_evaluation_datasets("missing-tenant", "missing-project") == []
    assert service.get_evaluation_dataset("missing-tenant", "missing-project", str(uuid4())) is None
class FailingQdrant:
    def ensure_collection(self) -> None:
        raise RuntimeError("Qdrant unavailable")


class RecordingPostgres:
    def __init__(self) -> None:
        self.statuses: list[str] = []

    def get_or_create_tenant(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(id=uuid4())

    def get_or_create_project(self, tenant_id, name: str) -> SimpleNamespace:
        return SimpleNamespace(id=uuid4())

    def create_job(self, tenant_id, project_id, job_type, inputs) -> SimpleNamespace:
        return SimpleNamespace(id=uuid4(), tenant_id=tenant_id, project_id=project_id)

    def update_job(self, job_id, status, outputs=None, error=None, progress=100) -> None:
        self.statuses.append(status)


def test_ingestion_marks_job_failed_when_storage_initialization_fails() -> None:
    service = PersistentRagService.__new__(PersistentRagService)
    service.postgres = RecordingPostgres()
    service.qdrant = FailingQdrant()
    service.chunk_size = 100
    service.chunk_overlap = 10

    with pytest.raises(RuntimeError, match="Qdrant unavailable"):
        service.ingest_text("tenant", "project", "source", "content")

    assert service.postgres.statuses == ["FAILED"]


class RecoverablePostgres(RecordingPostgres):
    def __init__(self) -> None:
        super().__init__()
        self.tenant_id, self.project_id = uuid4(), uuid4()
        self.document_id, self.chunk_id = uuid4(), uuid4()
        self.point_id = None
        self.document_created = False

    def get_or_create_tenant(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(id=self.tenant_id)

    def get_or_create_project(self, tenant_id, name) -> SimpleNamespace:
        return SimpleNamespace(id=self.project_id)

    def create_document_if_new(self, *args):
        created = not self.document_created
        self.document_created = True
        return SimpleNamespace(id=self.document_id), created

    def create_chunk_if_new(self, *args):
        return SimpleNamespace(id=self.chunk_id, qdrant_point_id=self.point_id), self.point_id is None

    def link_document_chunk(self, *args) -> None:
        pass

    def set_chunk_qdrant_point(self, chunk_id, point_id) -> None:
        self.point_id = point_id

    def write_trace(self, *args, **kwargs) -> None:
        pass


class RetryQdrant:
    def __init__(self) -> None:
        self.points: set[str] = set()
        self.fail_once = True

    def ensure_collection(self) -> None:
        pass

    def point_exists(self, point_id) -> bool:
        return str(point_id) in self.points

    def upsert_chunk(self, point_id, *args) -> None:
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("temporary vector outage")
        self.points.add(str(point_id))


def test_duplicate_ingestion_repairs_chunk_missing_its_vector_point() -> None:
    service = PersistentRagService.__new__(PersistentRagService)
    service.postgres = RecoverablePostgres()
    service.qdrant = RetryQdrant()
    service.embedder = HashEmbedder(8)
    service.embedding_dimensions = 8
    service.chunk_size = 100
    service.chunk_overlap = 10

    with pytest.raises(RuntimeError, match="temporary vector outage"):
        service.ingest_text("tenant", "project", "source", "repair this")
    stored_point_id = service.postgres.point_id
    repaired = service.ingest_text("tenant", "project", "source", "repair this")

    assert repaired["status"] == "completed"
    assert repaired["chunks_created"] == 1
    assert str(stored_point_id) in service.qdrant.points
