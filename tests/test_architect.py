import importlib
from pathlib import Path

import yaml

from mcp_servers.inspector import server
from mcp_servers.inspector.connectors import inspect_repository_integrations
import pytest
from rag_core.storage.postgres_adapter import PostgresAdapter


def test_recommendation_uses_requested_vector_store() -> None:
    recommendation = server.recommend_rag_stack("starter", "pgvector")
    assert recommendation["vector_store"] == "PostgreSQL with pgvector"


def test_project_plan_is_read_only() -> None:
    plan = server.create_rag_project_plan("Build a PDF question-answering service")
    assert "does not change" in plan["safety"]
    assert len(plan["implementation_steps"]) == 6


def test_destructive_tool_requires_confirmation() -> None:
    with pytest.raises(ValueError, match="confirm=True"):
        server.rag_delete(document_ids=["document-id"])


def test_content_hash_is_stable_and_metadata_sensitive() -> None:
    first = PostgresAdapter.content_hash("same text", {"page": 1})
    assert first == PostgresAdapter.content_hash("same text", {"page": 1})
    assert first != PostgresAdapter.content_hash("same text", {"page": 2})


def test_integration_inventory_detects_services_without_exposing_env_values(tmp_path) -> None:
    (tmp_path / "compose.yaml").write_text(
        "services:\n  vector-db:\n    image: qdrant/qdrant:latest\n"
        "    ports:\n      - '6333:6333'\n    environment:\n      API_KEY: do-not-return-this\n",
        encoding="utf-8",
    )
    (tmp_path / "package.json").write_text('{"scripts":{"test":"jest"}}', encoding="utf-8")

    inventory = inspect_repository_integrations(tmp_path)

    assert inventory["services"]["entries"][0]["connectors"] == ["docker", "qdrant"]
    assert inventory["services"]["entries"][0]["environment_keys"] == ["API_KEY"]
    assert inventory["tests"]["suggested_commands"] == ["npm test"]
    assert inventory["tests"]["executed"] is False
    assert "do-not-return-this" not in str(inventory)


def test_compose_waits_for_api_readiness_before_starting_public_proxy() -> None:
    root = Path(__file__).resolve().parents[1]
    compose = yaml.safe_load((root / "docker-compose.yml").read_text(encoding="utf-8"))
    assert "/readyz" in compose["services"]["api"]["healthcheck"]["test"][-1]
    assert compose["services"]["caddy"]["depends_on"]["api"]["condition"] == "service_healthy"
    assert compose["services"]["api"]["logging"]["driver"] == "json-file"
    assert all(service["restart"] == "unless-stopped" for service in compose["services"].values())


def test_container_build_excludes_secrets_and_runs_unprivileged() -> None:
    root = Path(__file__).resolve().parents[1]
    ignored = (root / ".dockerignore").read_text(encoding="utf-8").splitlines()
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    assert ".env" in ignored
    assert ".venv" in ignored
    assert "backups" in ignored
    assert "USER 10001:10001" in dockerfile
    assert "PYTHONDONTWRITEBYTECODE=1" in dockerfile


def test_public_proxy_sets_baseline_security_headers() -> None:
    root = Path(__file__).resolve().parents[1]
    caddyfile = (root / "Caddyfile").read_text(encoding="utf-8")
    for header in ("Strict-Transport-Security", "X-Content-Type-Options", "X-Frame-Options", "Referrer-Policy"):
        assert header in caddyfile


def test_trace_index_migration_is_idempotent_for_metadata_created_schema(monkeypatch) -> None:
    migration = importlib.import_module("rag_core.storage.migrations.versions.0003_trace_lookup_index")
    calls = []
    monkeypatch.setattr(migration.op, "create_index", lambda *args, **kwargs: calls.append((args, kwargs)))

    migration.upgrade()

    assert calls[0][0] == ("ix_traces_tenant_created_at", "traces", ["tenant_id", "created_at"])
    assert calls[0][1]["if_not_exists"] is True


def test_initial_migration_declares_its_schema_without_live_orm_metadata(monkeypatch) -> None:
    migration = importlib.import_module("rag_core.storage.migrations.versions.0001_initial_schema")
    tables, indexes = [], []
    monkeypatch.setattr(migration.op, "create_table", lambda name, *args, **kwargs: tables.append(name))
    monkeypatch.setattr(migration.op, "create_index", lambda name, table, *args, **kwargs: indexes.append((name, table)))

    migration.upgrade()

    assert tables == ["tenants", "projects", "documents", "chunks", "document_chunks", "jobs", "traces"]
    assert ("ix_traces_tenant_created_at", "traces") in indexes
    assert not hasattr(migration, "Base")
