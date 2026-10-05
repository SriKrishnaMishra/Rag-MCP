import hashlib
import json
import subprocess
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from scripts.backup import _copy_offsite, create_backup, verify_backup


def test_backup_writes_postgres_dump_qdrant_snapshot_and_hash_manifest(tmp_path, monkeypatch):
    def run(command, **kwargs):
        assert command[:5] == ["docker", "compose", "exec", "-T", "postgres"]
        kwargs["stdout"].write(b"PGDMPpostgres dump")
        return SimpleNamespace(returncode=0, stderr=b"")

    class Response:
        def __init__(self, body):
            self.body = body
            self.position = 0
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            return None
        def read(self, size=-1):
            if size < 0:
                return self.body
            chunk = self.body[self.position:self.position + size]
            self.position += len(chunk)
            return chunk

    def open_url(request, timeout):
        assert timeout == 20
        if request.get_method() == "POST":
            return Response(b'{"result":{"name":"snap-1.snapshot"}}')
        return Response(b"qdrant snapshot")

    monkeypatch.setattr("scripts.backup.subprocess.run", run)
    monkeypatch.setattr("scripts.backup.urlopen", open_url)
    result = create_backup(tmp_path, "rag_chunks", "http://127.0.0.1:6333", timeout=20)
    directory = Path(result["directory"])
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))

    assert (directory / "postgres.dump").read_bytes() == b"PGDMPpostgres dump"
    assert (directory / "qdrant-rag_chunks.snapshot").read_bytes() == b"qdrant snapshot"
    assert manifest["files"]["postgres.dump"]["sha256"] == hashlib.sha256(b"PGDMPpostgres dump").hexdigest()
    assert manifest["files"]["qdrant-rag_chunks.snapshot"]["sha256"] == hashlib.sha256(b"qdrant snapshot").hexdigest()
    assert verify_backup(directory)["verified"] is True
    if os.name != "nt":
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
        assert all(stat.S_IMODE((directory / name).stat().st_mode) == 0o600
                   for name in ("postgres.dump", "qdrant-rag_chunks.snapshot", "manifest.json"))


def test_backup_verification_detects_corrupted_dump(tmp_path, monkeypatch):
    def run(_command, **kwargs):
        kwargs["stdout"].write(b"PGDMPdatabase")
        return SimpleNamespace(returncode=0, stderr=b"")

    class Response:
        def __init__(self, body): self.body, self.position = body, 0
        def __enter__(self): return self
        def __exit__(self, *_args): return None
        def read(self, size=-1):
            if size < 0: return self.body
            chunk = self.body[self.position:self.position + size]
            self.position += len(chunk)
            return chunk

    def open_url(request, timeout):
        response = (b'{"result":{"name":"snap.snapshot"}}' if request.get_method() == "POST"
                    else b"qdrant snapshot")
        return Response(response)

    monkeypatch.setattr("scripts.backup.subprocess.run", run)
    monkeypatch.setattr("scripts.backup.urlopen", open_url)
    directory = Path(create_backup(tmp_path, "rag_chunks", "http://localhost:6333")["directory"])
    with (directory / "postgres.dump").open("ab") as dump:
        dump.write(b"corruption")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_backup(directory)


def test_failed_backup_removes_incomplete_staging_directory(tmp_path, monkeypatch):
    def run(_command, **kwargs):
        kwargs["stdout"].write(b"partial dump")
        return SimpleNamespace(returncode=1, stderr=b"database unavailable")

    monkeypatch.setattr("scripts.backup.subprocess.run", run)
    with pytest.raises(RuntimeError, match="PostgreSQL dump failed"):
        create_backup(tmp_path, "rag_chunks", "http://127.0.0.1:6333")
    assert list(tmp_path.iterdir()) == []


def test_offsite_copy_uses_immutable_rclone_copy_without_shell(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()

    def run(command, **kwargs):
        assert command == ["rclone", "copy", "--immutable", str(bundle), "encrypted:rag-backups"]
        assert kwargs["shell"] is False
        assert kwargs["stdout"] is subprocess.DEVNULL
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr("scripts.backup.subprocess.run", run)
    _copy_offsite(bundle, "encrypted:rag-backups", 30)


def test_offsite_copy_failure_retains_local_bundle(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    monkeypatch.setattr("scripts.backup.subprocess.run",
                        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stderr=b"secret-bearing error"))

    with pytest.raises(RuntimeError, match="local backup is retained"):
        _copy_offsite(bundle, "encrypted:rag-backups", 30)


def test_failed_offsite_transfer_keeps_published_local_backup(tmp_path, monkeypatch):
    class Response:
        def __init__(self, body): self.body, self.position = body, 0
        def __enter__(self): return self
        def __exit__(self, *_args): return None
        def read(self, size=-1):
            if size < 0: return self.body
            chunk = self.body[self.position:self.position + size]
            self.position += len(chunk)
            return chunk

    def run(command, **kwargs):
        if command[0] == "docker":
            kwargs["stdout"].write(b"PGDMPdatabase")
            return SimpleNamespace(returncode=0, stderr=b"")
        return SimpleNamespace(returncode=1, stderr=b"private remote details")

    def open_url(request, timeout):
        body = (b'{"result":{"name":"snapshot"}}' if request.get_method() == "POST"
                else b"qdrant snapshot")
        return Response(body)

    monkeypatch.setattr("scripts.backup.subprocess.run", run)
    monkeypatch.setattr("scripts.backup.urlopen", open_url)
    with pytest.raises(RuntimeError, match="Local backup completed") as error:
        create_backup(tmp_path, "rag_chunks", "http://localhost:6333",
                      offsite_destination="encrypted:rag-backups")

    backups = list(tmp_path.glob("rag-backup-*"))
    assert len(backups) == 1
    assert (backups[0] / "manifest.json").is_file()
    assert "private remote details" not in str(error.value)


@pytest.mark.parametrize("collection", ["../unsafe", "has space", ""])
def test_backup_rejects_unsafe_collection_name(tmp_path, collection):
    with pytest.raises(ValueError, match="collection name"):
        create_backup(tmp_path, collection, "http://127.0.0.1:6333")
