# MCP-Powered RAG Engineering Agent

A tenant-aware RAG backend with FastAPI, PostgreSQL, Qdrant, Celery, Ollama, and MCP inspection tools.

## What works now

- FastAPI API for text and file ingestion (TXT, Markdown, PDF, DOCX), document listing, metadata-filtered search, grounded answers, and readiness checks.
- Small dependency-free retrieval engine, so the demo runs without external services.
- MCP inspection server with project, route, dependency, Docker-service/test integration inventory, RAG-pipeline, and log tools.
- Reusable RAG Project Architect tools that can inspect another repository selected with `TARGET_REPOSITORY`.
- Collection-aware RAG MCP operations with traces, evaluation metrics, dense plus lexical hybrid search, and confirmation-gated in-memory deletion/reindex requests.
- Tenant-scoped PostgreSQL/Qdrant persistence with idempotent ingestion, queued Celery jobs, and job status lookup.
- Unit tests for the in-memory API and core behavior, plus Docker Compose for the persistent stack.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs`, then use `POST /documents` followed by `POST /search`.

`POST /documents/upload` accepts TXT, Markdown, PDF, and DOCX multipart uploads. Direct document ingestion is capped at 5 million characters and search/ask queries at 4,000 characters. Metadata and filter maps allow at most 32 entries, with keys capped at 128 characters and values at 512; per-request Ollama model names are capped at 128 characters. Search and ask source results include document metadata. `GET /documents` supports newest-first pagination with `limit` (1–500, default 100) and `offset` (0–1,000,000); evaluation list endpoints use the same bounds. Listing an unknown tenant or project returns an empty list without creating database records. In persistent mode, `POST /jobs/ingest` queues work and returns a job ID; poll `GET /jobs/{job_id}` for status. `GET /readyz` checks PostgreSQL, Qdrant, and Redis connectivity.

Persistent evaluation endpoints support saved datasets of up to 100 cases and run history: create with `POST /evaluation/datasets`, list with `GET /evaluation/datasets`, run with `POST /evaluation/datasets/{id}/runs`, and review prior runs with `GET /evaluation/datasets/{id}/runs`. Dataset and run lists support `limit` (1–500, default 100) and `offset` pagination.

Run the tests:

```powershell
pytest
```

Start the MCP server (stdio transport):

```powershell
python -m mcp_servers.inspector.server
```

Inspect another repository; edits and test runs require separate confirmation:

```powershell
$env:TARGET_REPOSITORY = "D:\path\to\another-repository"
python -m mcp_servers.inspector.server
```

## MCP tools

| Tool | Purpose |
| --- | --- |
| `inspect_project` | Show project structure and key files. |
| `read_source_file` | Read a safe, workspace-relative source file. |
| `search_code` | Search source code. |
| `inspect_routes` | List FastAPI routes. |
| `inspect_dependencies` | Show declared dependencies. |
| `inspect_integrations` | Inventory selected-repository Docker services and test commands without running repository code or revealing environment values. |
| `inspect_docker_runtime_status` | Read the selected repository's current Compose service status; never starts or changes services. |
| `probe_selected_repository_dependencies` | With `confirm=true`, run a credential-free PostgreSQL startup handshake, Redis `PING`, and Qdrant `/healthz` on explicitly loopback-published Compose ports. |
| `run_selected_repository_tests` | Run detected pytest/Jest/Vitest tests only with `confirm=true`, fixed command arguments, a sanitized environment, output redaction, and a timeout. Test code can still perform arbitrary actions as the OS user, so only run trusted repositories. |
| `rag_prepare_source_edit` / `rag_apply_approved_source_edit` | Review an exact source diff and hash, then separately apply that same preview with `confirm=true` if the file has not changed. |
| `rag_check_dependencies` | Check persistent PostgreSQL, Qdrant, and Redis connectivity. |
| `rag_inspect_pipeline` | Describe RAG configuration and collection state. |
| `rag_search` | Test retrieval without modifying data. |
| `inspect_logs` | Describe local Docker log retention and the operator command; it does not return log contents. |
| `detect_rag_stack` | Detect RAG libraries and services in the selected repository. |
| `recommend_rag_stack` | Recommend a starter, PostgreSQL, or enterprise RAG stack. |
| `create_rag_project_plan` | Generate an implementation plan without modifying a repository. |
| `rag_list_collections` / `rag_collection_info` | Inspect collection configuration and statistics. |
| `rag_ingest_text` / `rag_ingest_file` | Ingest text or restricted `.txt`/`.md` files. |
| `rag_search` / `rag_eval_query` / `rag_eval_dataset` | Search a collection and evaluate individual or batched queries. |
| `rag_delete` / `rag_reindex_collection` | Confirmation-gated deletion and reindexing. Persistent deletion scopes IDs to the selected tenant/project and removes Qdrant points before PostgreSQL rows; reindexing recomputes vectors with the configured embedder. The in-memory backend has no materialized index, so reindex is a no-op explanation there. |
| `rag_last_traces` | Inspect recent structured tool traces. |
| `rag_hybrid_search` / `rag_get_job` | Persistent tenant-isolated search and job lookup (requires persistent mode). |

MCP ingestion and retrieval tools enforce the same main bounds as the HTTP API: text up to 5 million characters, queries up to 4,000, 32 metadata entries with 128-character keys and 512-character string values, and collection names limited to 64 safe characters. File ingestion checks the encoded file size before reading and caps decoded text at 5 million characters. MCP single-query and batch evaluation support metadata filters and expected document IDs. Confirmed deletion batches accept at most 500 document/chunk IDs, each 1–128 characters long.

Repository inventory prunes hidden and generated directories, scans at most 20,000 files or 20,000 directories, and reports when it truncates. Individual source reads are capped at 2 MB; code search and route inspection skip larger files and scan at most 25 MB across 5,000 files per call. Search terms are limited to 256 characters, code-search matches to 100, and inspected routes to 500. Stack detection reads at most 500 small dependency/configuration files under a 10 MB aggregate limit.

## Persistent deployment

Pushes and pull requests run the Python 3.12 CI workflow, which installs the editable package, checks dependency consistency, and runs the test suite.

The repository includes PostgreSQL metadata storage, Qdrant vector storage, Alembic schema setup, and a Celery/Redis worker. Start the local stack with:

```powershell
docker compose up --build
```

Compose binds published ports to loopback, waits for its dependencies, applies Alembic migrations before starting the API, and restarts services unless explicitly stopped. Its API healthcheck polls `/readyz`, and the optional Caddy proxy waits for the API to become healthy. The API and worker image runs as an unprivileged user; `.dockerignore` excludes local credentials, backups, virtual environments, and tests from the build context. The Qdrant adapter creates the configured collection and indexes tenant/project fields. Persistent vector searches always require both tenant and project scope. The bundled `change-me` database credential is for local development only; set `POSTGRES_PASSWORD` (and a matching URL-encoded `DATABASE_URL` if needed) before deployment. Keep `.env` out of version control.

For an externally managed Qdrant service, set `QDRANT_URL` and (if required) `QDRANT_API_KEY` through the deployment secret store; Compose passes both to the API and worker. The bundled local Qdrant service remains the default and is published on loopback without API-key enforcement.

Set `RAG_STORAGE_BACKEND=persistent` to use PostgreSQL and Qdrant. The default hash embedder is deterministic and intended for local wiring. `EMBEDDING_PROVIDER=ollama` uses the configured Ollama embedding model; set `EMBEDDING_DIMENSIONS` to match that model and the Qdrant collection.

When running the API directly on the host with the Compose services, change `DATABASE_URL` to use `localhost` instead of the Compose service name `postgres`.

### Answers and background jobs

Install Ollama and pull the configured chat model to enable `POST /ask`. Persistent `POST /jobs/ingest` queues ingestion through Redis and Celery; use `GET /jobs/{job_id}` to check completion. Celery accepts and emits JSON task payloads only. The worker acknowledges jobs after execution, requeues work when a worker process is lost, and reserves one task per worker process; Redis broker and result backend use a six-hour visibility timeout. These settings rely on ingestion's content-hash idempotency so a worker crash can safely cause redelivery. The worker retries transient connection, timeout, PostgreSQL operational, HTTP transport, Qdrant transport, and Qdrant/Ollama 429/5xx failures with exponential backoff (up to three retries); client/configuration errors are not retried.

### Authentication and deployment limits

Set `AUTH_REQUIRED=true` and provide `TENANT_API_KEYS` as a non-empty JSON map from API key to tenant ID through a secret manager; startup rejects authentication mode with an empty map. Boolean environment flags accept `true`/`false`, `1`/`0`, `yes`/`no`, and `on`/`off`; other values fail startup to catch misspellings. Set `RATE_LIMIT_PER_MINUTE` to a positive value to enable Redis-backed API rate limiting; it fails closed when Redis is unavailable unless `RATE_LIMIT_FAIL_OPEN=true`. The default is disabled for local development. Compose publishes services on loopback; use the supplied Caddy profile or another TLS-terminating proxy before public access. API responses include a generated `X-Request-ID`; request logs record that ID, method, status, duration, and on failures only the exception class, not request bodies, query strings, or exception text. Persistent RAG traces are stored in PostgreSQL and indexed for tenant-scoped newest-first retrieval; search traces retain query length and result counts, not raw query text. Secret rotation, durable log/trace retention, off-host backup storage, and organization-grade authentication still require deployment-specific setup. The API-key mechanism is a starter, not SSO/OAuth or role-based access control.

### Persistent data backups

With the Compose stack running and Qdrant reachable at `QDRANT_URL`, quiesce API writes and ingestion workers, then run:

```powershell
python scripts/backup.py --output-dir backups
```

The script writes a PostgreSQL custom-format logical dump, a Qdrant collection snapshot, and a SHA-256 manifest to a timestamped directory. On POSIX hosts the bundle directory is mode `0700` and its files are mode `0600`; on Windows, protect the bundle with the destination directory's ACL. It stages each run under a hidden incomplete name and publishes the final directory only after all files and the manifest are written; failed runs remove their local staging directory. Set `QDRANT_COLLECTION`, `QDRANT_URL`, and (when Qdrant authentication is enabled) `QDRANT_API_KEY` in the environment. Optional off-site copying uses `rclone copy --immutable`; set `BACKUP_RCLONE_DESTINATION` to an rclone remote path, and configure that remote as an rclone `crypt` remote to encrypt file contents and names client-side. A failed copy reports failure while retaining the completed local bundle. Configure remote retention separately. Backup contents are ignored by Git. Pause writes across both stores to obtain a consistent recovery point. Restore the PostgreSQL dump with `pg_restore` into a prepared database, and restore the Qdrant snapshot using Qdrant's [collection snapshot recovery API](https://api.qdrant.tech/api-reference/snapshots/recover-from-uploaded-snapshot); both restore operations overwrite target data and should be exercised against a disposable environment before production use.

For Linux hosts using a user-level systemd manager, `deploy/rag-backup.service` and `deploy/rag-backup.timer` provide a daily 02:00 host-local-time schedule (with up to 20 minutes of randomized delay and catch-up after downtime). Install both under `~/.config/systemd/user/`, adjust the checkout path in the service if it is not `~/mcp-backend-rag`, create `~/.config/mcp-rag/backup.env` with Qdrant settings and optionally `BACKUP_RCLONE_DESTINATION=encrypted-remote:rag-backups`, then run `systemctl --user daemon-reload` and `systemctl --user enable --now rag-backup.timer`. The systemd user must be able to run Docker Compose against the project and reach Qdrant on the configured URL; install and configure rclone for the same user when off-site copying is enabled. For backups to run while logged out, enable lingering for that user and ensure its Docker daemon is available. Check executions with `systemctl --user list-timers rag-backup.timer` and `journalctl --user -u rag-backup.service`.

After copying or before restoring a bundle, verify its file sizes, SHA-256 checksums, and PostgreSQL custom-dump signature with:

```powershell
python scripts/backup.py --verify backups/rag-backup-<timestamp>
```

For an optional public HTTPS entrypoint, set `CADDY_DOMAIN` to a DNS name pointing at the host, allow inbound ports 80 and 443, then start with `docker compose --profile public up --build`. Caddy provisions and renews certificates, adds baseline HSTS/content-type/frame/referrer headers, and its network can reach the API but not the database, vector store, or Redis directly. Enable API authentication and rate limiting before using this profile publicly.

## Remaining work

- A model-driven MCP tool selector. It needs an explicit policy for sending user queries to the configured Ollama URL, since that URL can be remote.
- The cross-repository dependency probe does not validate PostgreSQL credentials/schema or authenticated Redis access. Qdrant health uses its unauthenticated local `/healthz` endpoint.
- Backup scheduling and encrypted off-site transfer have host-side templates, but deployment still must install/enable the systemd timer, configure an rclone `crypt` remote and remote retention, and confirm transfers from the deployment host.
- Production deployment still needs restore drills against disposable PostgreSQL/Qdrant targets, secret rotation, durable log/trace retention, and organization-grade authentication.
