"""PostgreSQL adapter. It is the transaction boundary and system of record."""
from __future__ import annotations

from hashlib import sha256
import json
from uuid import UUID
import re

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from rag_core.storage.models import (Base, Chunk, Document, DocumentChunk, EvaluationCase,
                                     EvaluationDataset, EvaluationRun, Job, Project, Tenant, Trace)


class PostgresAdapter:
    def __init__(self, database_url: str) -> None:
        self.engine = create_engine(database_url, pool_pre_ping=True)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    def create_schema_for_local_development(self) -> None:
        """For local use only. Production deploys schema through Alembic."""
        Base.metadata.create_all(self.engine)

    def get_or_create_tenant(self, name: str, config: dict[str, object] | None = None) -> Tenant:
        with self.sessions.begin() as session:
            tenant = session.scalar(select(Tenant).where(Tenant.name == name))
            if tenant is None:
                tenant = Tenant(name=name, config=config or {})
                session.add(tenant)
                session.flush()
            return tenant

    def get_tenant(self, name: str) -> Tenant | None:
        with Session(self.engine) as session:
            return session.scalar(select(Tenant).where(Tenant.name == name))

    def ping(self) -> bool:
        from sqlalchemy import text
        with self.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True

    def create_project(self, tenant_id: UUID, name: str, config: dict[str, object] | None = None) -> Project:
        with self.sessions.begin() as session:
            project = Project(tenant_id=tenant_id, name=name, config=config or {})
            session.add(project)
            session.flush()
            return project

    def get_or_create_project(self, tenant_id: UUID, name: str, config: dict[str, object] | None = None) -> Project:
        with self.sessions.begin() as session:
            project = session.scalar(select(Project).where(Project.tenant_id == tenant_id, Project.name == name))
            if project is None:
                project = Project(tenant_id=tenant_id, name=name, config=config or {})
                session.add(project)
                session.flush()
            return project

    def get_project(self, tenant_id: UUID, name: str) -> Project | None:
        with Session(self.engine) as session:
            return session.scalar(select(Project).where(Project.tenant_id == tenant_id, Project.name == name))

    @staticmethod
    def content_hash(text: str, metadata: dict[str, object] | None = None, namespace: str = "") -> str:
        stable_metadata = json.dumps(metadata or {}, sort_keys=True, separators=(",", ":"), default=str)
        return sha256(f"{namespace}|{text.strip()}|{stable_metadata}".encode("utf-8")).hexdigest()

    def create_document_if_new(self, tenant_id: UUID, project_id: UUID, source_type: str, source_path: str,
                               mime_type: str, content: str, metadata: dict[str, object] | None = None) -> tuple[Document, bool]:
        content_hash = self.content_hash(content, metadata, str(project_id))
        with self.sessions.begin() as session:
            existing = session.scalar(select(Document).where(Document.tenant_id == tenant_id,
                                                               Document.content_hash == content_hash))
            if existing:
                return existing, False
            document = Document(tenant_id=tenant_id, project_id=project_id, source_type=source_type,
                                source_path=source_path, mime_type=mime_type, content_hash=content_hash,
                                metadata_json=metadata or {})
            session.add(document)
            session.flush()
            return document, True

    def create_chunk_if_new(self, tenant_id: UUID, document_id: UUID, chunk_index: int, text: str,
                            metadata: dict[str, object] | None = None) -> tuple[Chunk, bool]:
        content_hash = self.content_hash(text, metadata, f"{document_id}:{chunk_index}")
        with self.sessions.begin() as session:
            existing = session.scalar(select(Chunk).where(Chunk.tenant_id == tenant_id, Chunk.content_hash == content_hash))
            if existing:
                return existing, False
            chunk = Chunk(tenant_id=tenant_id, document_id=document_id, chunk_index=chunk_index, text=text,
                          content_hash=content_hash, metadata_json=metadata or {})
            session.add(chunk)
            session.flush()
            return chunk, True

    def link_document_chunk(self, document_id: UUID, chunk_id: UUID) -> None:
        with self.sessions.begin() as session:
            link = session.get(DocumentChunk, {"document_id": document_id, "chunk_id": chunk_id})
            if link is None:
                session.add(DocumentChunk(document_id=document_id, chunk_id=chunk_id))

    def set_chunk_qdrant_point(self, chunk_id: UUID, point_id: UUID) -> None:
        with self.sessions.begin() as session:
            chunk = session.get(Chunk, chunk_id)
            if chunk:
                chunk.qdrant_point_id = point_id

    def resolve_delete_targets(self, tenant_id: UUID, project_id: UUID,
                               document_ids: list[UUID], chunk_ids: list[UUID]) -> dict[str, object]:
        """Resolve requested IDs under a project scope before touching vector storage."""
        with Session(self.engine) as session:
            documents = session.scalars(select(Document).where(
                Document.tenant_id == tenant_id, Document.project_id == project_id,
                Document.id.in_(document_ids) if document_ids else False
            )).all()
            chunks = session.scalars(select(Chunk).join(
                Document, Chunk.document_id == Document.id).where(
                Chunk.tenant_id == tenant_id, Document.project_id == project_id,
                Chunk.id.in_(chunk_ids) if chunk_ids else False
            )).all()
            document_chunk_rows = session.execute(select(Chunk).join(
                Document, Chunk.document_id == Document.id).where(
                Chunk.tenant_id == tenant_id, Document.tenant_id == tenant_id,
                Document.project_id == project_id,
                Document.id.in_([item.id for item in documents]) if documents else False
            )).scalars().all()
            all_chunks = {item.id: item for item in [*chunks, *document_chunk_rows]}
            return {"document_ids": [item.id for item in documents],
                    "chunk_ids": [item.id for item in chunks],
                    "all_chunk_ids": list(all_chunks),
                    "point_ids": [item.qdrant_point_id for item in all_chunks.values()
                                  if item.qdrant_point_id is not None]}

    def delete_resolved_targets(self, tenant_id: UUID, project_id: UUID,
                                targets: dict[str, object]) -> dict[str, int]:
        document_ids = list(targets["document_ids"])
        chunk_ids = list(targets["chunk_ids"])
        all_chunk_ids = list(targets["all_chunk_ids"])
        with self.sessions.begin() as session:
            documents = session.scalars(select(Document).where(
                Document.tenant_id == tenant_id, Document.project_id == project_id,
                Document.id.in_(document_ids) if document_ids else False
            )).all()
            chunks = session.scalars(select(Chunk).join(
                Document, Chunk.document_id == Document.id).where(
                Chunk.tenant_id == tenant_id, Document.project_id == project_id,
                Chunk.id.in_(chunk_ids) if chunk_ids else False
            )).all()
            document_count, chunk_count = len(documents), len(all_chunk_ids)
            for document in documents:
                session.delete(document)
            for chunk in chunks:
                session.delete(chunk)
            return {"documents_deleted": document_count, "chunks_deleted": chunk_count}

    def list_chunks_for_project(self, tenant_id: UUID, project_id: UUID) -> list[dict[str, object]]:
        with Session(self.engine) as session:
            rows = session.execute(select(Chunk, Document.source_path).join(
                Document, Chunk.document_id == Document.id).where(
                Chunk.tenant_id == tenant_id, Document.tenant_id == tenant_id,
                Document.project_id == project_id).order_by(Document.id, Chunk.chunk_index)).all()
            return [{"chunk_id": chunk.id, "document_id": chunk.document_id,
                     "point_id": chunk.qdrant_point_id, "chunk_index": chunk.chunk_index,
                     "text": chunk.text, "source": source, "metadata": chunk.metadata_json}
                    for chunk, source in rows]


    def create_job(self, tenant_id: UUID, project_id: UUID, job_type: str, inputs: dict[str, object]) -> Job:
        with self.sessions.begin() as session:
            job = Job(tenant_id=tenant_id, project_id=project_id, type=job_type, inputs=inputs)
            session.add(job)
            session.flush()
            return job

    def get_job(self, job_id: UUID) -> Job | None:
        with Session(self.engine) as session:
            return session.get(Job, job_id)

    def get_job_for_tenant(self, job_id: UUID, tenant_id: UUID) -> Job | None:
        with Session(self.engine) as session:
            return session.scalar(select(Job).where(Job.id == job_id, Job.tenant_id == tenant_id))

    def list_documents(self, tenant_id: UUID, project_id: UUID,
                       limit: int = 100, offset: int = 0) -> list[dict[str, object]]:
        with Session(self.engine) as session:
            documents = session.scalars(select(Document).where(
                Document.tenant_id == tenant_id, Document.project_id == project_id
            ).order_by(Document.created_at.desc(), Document.id.desc()).offset(offset).limit(limit)).all()
            return [{"document_id": str(item.id), "title": item.source_path, "source": item.source_path,
                     "source_type": item.source_type, "mime_type": item.mime_type,
                     "metadata": item.metadata_json, "created_at": item.created_at.isoformat() if item.created_at else None}
                    for item in documents]

    def list_projects(self, tenant_id: UUID) -> list[dict[str, object]]:
        with Session(self.engine) as session:
            projects = session.scalars(select(Project).where(Project.tenant_id == tenant_id)
                                       .order_by(Project.name)).all()
            output = []
            for project in projects:
                documents = session.scalar(select(func.count()).select_from(Document).where(
                    Document.tenant_id == tenant_id, Document.project_id == project.id)) or 0
                chunks = session.scalar(select(func.count()).select_from(Chunk).where(
                    Chunk.tenant_id == tenant_id).join(Document, Chunk.document_id == Document.id)
                                        .where(Document.project_id == project.id)) or 0
                output.append({"name": project.name, "documents": documents, "chunks": chunks,
                               "config": project.config})
            return output

    def list_traces(self, tenant_id: UUID, limit: int = 20) -> list[dict[str, object]]:
        with Session(self.engine) as session:
            traces = session.scalars(select(Trace).where(Trace.tenant_id == tenant_id)
                                     .order_by(Trace.created_at.desc()).limit(limit)).all()
            return [{"id": str(item.id), "type": item.type, "tool_name": item.tool_name,
                     "timestamp": item.created_at.isoformat() if item.created_at else None,
                     "inputs": self._sanitize_trace_inputs(item.inputs),
                     "outputs": item.outputs, "latency_ms": item.latency_ms,
                     "error": item.error} for item in traces]

    @staticmethod
    def _sanitize_trace_inputs(inputs: object) -> dict[str, object]:
        """Redact raw query text from historical traces when they are exposed to callers."""
        if not isinstance(inputs, dict):
            return {}
        sanitized = dict(inputs)
        raw_query = sanitized.pop("query", None)
        if isinstance(raw_query, str):
            sanitized["query_length"] = len(raw_query)
        return sanitized

    @staticmethod
    def _dataset_dict(dataset: EvaluationDataset, cases: list[EvaluationCase] | None = None) -> dict[str, object]:
        return {"id": str(dataset.id), "name": dataset.name, "description": dataset.description,
                "created_at": dataset.created_at.isoformat() if dataset.created_at else None,
                "cases": [{"id": str(case.id), "position": case.position, "query": case.query,
                           "expected_answer": case.expected_answer,
                           "expected_document_id": case.expected_document_id,
                           "metadata": case.metadata_json} for case in (cases or [])]}

    def create_evaluation_dataset(self, tenant_id: UUID, project_id: UUID, name: str, description: str,
                                  cases: list[dict[str, object]]) -> dict[str, object]:
        with self.sessions.begin() as session:
            exists = session.scalar(select(EvaluationDataset.id).where(
                EvaluationDataset.tenant_id == tenant_id, EvaluationDataset.project_id == project_id,
                EvaluationDataset.name == name))
            if exists:
                raise ValueError("an evaluation dataset with this name already exists in the project")
            dataset = EvaluationDataset(tenant_id=tenant_id, project_id=project_id,
                                        name=name, description=description)
            session.add(dataset)
            session.flush()
            rows = [EvaluationCase(dataset_id=dataset.id, position=index, query=str(item["query"]),
                                   expected_answer=str(item.get("expected_answer", "")),
                                   expected_document_id=(str(item["expected_document_id"])
                                                         if item.get("expected_document_id") else None),
                                   metadata_json=dict(item.get("metadata", {})))
                    for index, item in enumerate(cases)]
            session.add_all(rows)
            session.flush()
            return self._dataset_dict(dataset, rows)

    def list_evaluation_datasets(self, tenant_id: UUID, project_id: UUID,
                                 limit: int = 100, offset: int = 0) -> list[dict[str, object]]:
        with Session(self.engine) as session:
            datasets = session.scalars(select(EvaluationDataset).where(
                EvaluationDataset.tenant_id == tenant_id, EvaluationDataset.project_id == project_id
            ).order_by(EvaluationDataset.created_at.desc(), EvaluationDataset.id.desc())
                .offset(offset).limit(limit)).all()
            return [self._dataset_dict(item) for item in datasets]

    def get_evaluation_dataset(self, tenant_id: UUID, project_id: UUID,
                               dataset_id: UUID) -> dict[str, object] | None:
        with Session(self.engine) as session:
            dataset = session.scalar(select(EvaluationDataset).where(
                EvaluationDataset.tenant_id == tenant_id, EvaluationDataset.project_id == project_id,
                EvaluationDataset.id == dataset_id))
            if dataset is None:
                return None
            cases = session.scalars(select(EvaluationCase).where(
                EvaluationCase.dataset_id == dataset.id).order_by(EvaluationCase.position)).all()
            return self._dataset_dict(dataset, cases)

    def create_evaluation_run(self, tenant_id: UUID, project_id: UUID, dataset_id: UUID,
                              metrics: dict[str, object], results: list[dict[str, object]]) -> dict[str, object]:
        with self.sessions.begin() as session:
            run = EvaluationRun(tenant_id=tenant_id, project_id=project_id, dataset_id=dataset_id,
                                metrics=metrics, results=results)
            session.add(run)
            session.flush()
            return {"id": str(run.id), "dataset_id": str(run.dataset_id), "metrics": run.metrics,
                    "results": run.results, "created_at": run.created_at.isoformat() if run.created_at else None}

    def list_evaluation_runs(self, tenant_id: UUID, project_id: UUID,
                             dataset_id: UUID, limit: int = 100, offset: int = 0) -> list[dict[str, object]]:
        with Session(self.engine) as session:
            runs = session.scalars(select(EvaluationRun).where(
                EvaluationRun.tenant_id == tenant_id, EvaluationRun.project_id == project_id,
                EvaluationRun.dataset_id == dataset_id).order_by(EvaluationRun.created_at.desc(), EvaluationRun.id.desc())
                .offset(offset).limit(limit)).all()
            return [{"id": str(run.id), "dataset_id": str(run.dataset_id), "metrics": run.metrics,
                     "results": run.results, "created_at": run.created_at.isoformat() if run.created_at else None}
                    for run in runs]

    def lexical_search(self, tenant_id: UUID, project_id: UUID, query: str, limit: int = 20) -> list[dict[str, object]]:
        terms = list(dict.fromkeys(re.findall(r"[a-zA-Z0-9_]+", query.lower())))[:12]
        if not terms:
            return []
        from sqlalchemy import or_
        predicates = [Chunk.text.ilike(f"%{term}%") for term in terms]
        with Session(self.engine) as session:
            rows = session.execute(select(Chunk, Document.source_path).join(
                Document, Chunk.document_id == Document.id).where(
                Chunk.tenant_id == tenant_id, Document.tenant_id == tenant_id,
                Document.project_id == project_id, or_(*predicates)
            ).limit(limit)).all()
            return [{"document_id": str(chunk.document_id), "chunk_id": str(chunk.id),
                     "title": source, "content": chunk.text, "text": chunk.text,
                     "metadata": chunk.metadata_json} for chunk, source in rows]

    def update_job(self, job_id: UUID, status: str, outputs: dict[str, object] | None = None,
                   error: str | None = None, progress: int = 100) -> None:
        with self.sessions.begin() as session:
            job = session.get(Job, job_id)
            if job is None:
                raise ValueError("job does not exist")
            job.status, job.outputs, job.error, job.progress = status, outputs or {}, error, progress

    def write_trace(self, tenant_id: UUID, project_id: UUID | None, trace_type: str, tool_name: str,
                    inputs: dict[str, object], outputs: dict[str, object], latency_ms: int | None = None,
                    error: str | None = None) -> Trace:
        with self.sessions.begin() as session:
            trace = Trace(tenant_id=tenant_id, project_id=project_id, type=trace_type, tool_name=tool_name,
                          inputs=inputs, outputs=outputs, latency_ms=latency_ms, error=error)
            session.add(trace)
            session.flush()
            return trace
