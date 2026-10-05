"""Create a logical PostgreSQL dump and downloadable Qdrant collection snapshot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.error import URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _restrict_permissions(path: Path, mode: int) -> None:
    """Keep backup data private on POSIX hosts; Windows uses its inherited ACL."""
    if os.name != "nt":
        os.chmod(path, mode)


def _create_private_file(path: Path):
    """Create a new backup file with owner-only permissions from its first byte."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(path, flags, 0o600)
    return os.fdopen(descriptor, "wb")


def _request(url: str, method: str, timeout: int, api_key: str | None = None) -> tuple[bytes, dict[str, object] | None]:
    headers = {"api-key": api_key} if api_key else {}
    request = Request(url, method=method, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        body = response.read()
    if method == "POST":
        parsed = json.loads(body)
        result = parsed.get("result") if isinstance(parsed, dict) else None
        if not isinstance(result, dict):
            raise RuntimeError("Qdrant snapshot API returned an unexpected response")
        return body, result
    return body, None


def _download(url: str, path: Path, timeout: int, api_key: str | None) -> None:
    headers = {"api-key": api_key} if api_key else {}
    with urlopen(Request(url, headers=headers), timeout=timeout) as response, path.open("wb") as target:
        while block := response.read(1024 * 1024):
            target.write(block)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _copy_offsite(directory: Path, destination: str, timeout: int) -> None:
    """Copy a completed bundle to a configured rclone remote without shell evaluation."""
    remote_name = destination.split(":", 1)[0] if ":" in destination else ""
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", remote_name)
            or len(destination) > 2048 or "\x00" in destination):
        raise ValueError("off-site destination must be an rclone remote path such as crypt-backups:rag")
    command = ["rclone", "copy", "--immutable", str(directory), destination]
    try:
        process = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                 timeout=timeout, cwd=REPOSITORY_ROOT,
                                 env=os.environ.copy(), check=False, shell=False)
    except FileNotFoundError as error:
        raise RuntimeError("rclone is not installed; local backup is retained") from error
    if process.returncode:
        raise RuntimeError(f"rclone failed; local backup is retained at {directory}")


def create_backup(output_dir: Path, collection: str, qdrant_url: str, timeout: int = 120,
                  offsite_destination: str | None = None) -> dict[str, object]:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", collection):
        raise ValueError("collection name may contain only letters, digits, underscores, and hyphens")
    if not 1 <= timeout <= 3600:
        raise ValueError("timeout must be between 1 and 3600 seconds")
    parsed = urlsplit(qdrant_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Qdrant URL must be an HTTP(S) URL without embedded credentials")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output_root = output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    final_dir = output_root / f"rag-backup-{timestamp}"
    destination = output_root / f".rag-backup-{timestamp}.incomplete"
    destination.mkdir(mode=0o700, exist_ok=False)
    _restrict_permissions(destination, 0o700)
    pg_path = destination / "postgres.dump"
    qdrant_path = destination / f"qdrant-{collection}.snapshot"
    env = os.environ.copy()
    # Expand the already configured database identity inside the running service.
    command = ["docker", "compose", "exec", "-T", "postgres", "sh", "-c",
               'exec pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc']
    try:
        with _create_private_file(pg_path) as target:
            process = subprocess.run(command, stdout=target, stderr=subprocess.PIPE, timeout=timeout,
                                     cwd=REPOSITORY_ROOT, env=env, check=False, shell=False)
        if process.returncode:
            raise RuntimeError("PostgreSQL dump failed; verify Docker Compose and database health")
        base = f"{parsed.scheme}://{parsed.netloc}"
        collection_path = quote(collection, safe="")
        api_key = env.get("QDRANT_API_KEY")
        _, snapshot = _request(f"{base}/collections/{collection_path}/snapshots?wait=true", "POST", timeout, api_key)
        assert snapshot is not None
        snapshot_name = snapshot.get("name")
        if not isinstance(snapshot_name, str) or Path(snapshot_name).name != snapshot_name:
            raise RuntimeError("Qdrant returned an invalid snapshot name")
        with _create_private_file(qdrant_path):
            pass
        _download(f"{base}/collections/{collection_path}/snapshots/{quote(snapshot_name, safe='')}",
                  qdrant_path, timeout, api_key)
        files = {}
        for path in (pg_path, qdrant_path):
            files[path.name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        manifest = {"created_at": timestamp, "collection": collection, "qdrant_snapshot": snapshot_name,
                    "postgres_dump_format": "custom", "files": files,
                    "consistency_note": "Quiesce API writes and ingestion workers before running for a cross-store consistent recovery point."}
        with _create_private_file(destination / "manifest.json") as manifest_file:
            manifest_file.write((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
        destination.rename(final_dir)
        result: dict[str, object] = {"directory": str(final_dir), "manifest": manifest}
        if offsite_destination:
            try:
                _copy_offsite(final_dir, offsite_destination, timeout)
            except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
                raise RuntimeError(f"Local backup completed at {final_dir}, but off-site copy failed: {error}") from error
            result["offsite"] = {"status": "uploaded", "destination": offsite_destination}
        return result
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def verify_backup(directory: Path) -> dict[str, object]:
    """Verify manifest paths, byte counts, checksums, and PostgreSQL custom-dump magic."""
    if directory.is_symlink():
        raise ValueError("backup path must be a directory, not a symlink")
    directory = directory.resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("backup path must be a directory, not a symlink")
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("manifest.json must be a regular file")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("backup manifest is not valid JSON") from error
    if not isinstance(manifest, dict) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", str(manifest.get("collection", ""))):
        raise ValueError("backup manifest is missing a valid collection")
    files = manifest.get("files")
    expected = {"postgres.dump", f"qdrant-{manifest['collection']}.snapshot"}
    if not isinstance(files, dict) or set(files) != expected:
        raise ValueError("backup manifest file list does not match the required bundle")
    checked: dict[str, object] = {}
    for name in sorted(expected):
        path = directory / name
        entry = files[name]
        if path.is_symlink() or not path.is_file() or not isinstance(entry, dict):
            raise ValueError(f"backup file is missing or invalid: {name}")
        size = path.stat().st_size
        if size < 1 or entry.get("bytes") != size or entry.get("sha256") != _sha256(path):
            raise ValueError(f"backup size or SHA-256 verification failed: {name}")
        checked[name] = {"bytes": size, "sha256": entry["sha256"]}
    with (directory / "postgres.dump").open("rb") as dump:
        if dump.read(5) != b"PGDMP":
            raise ValueError("postgres.dump is not a PostgreSQL custom-format dump")
    return {"directory": str(directory), "verified": True, "files": checked}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path, help="verify an existing backup bundle and exit")
    parser.add_argument("--output-dir", type=Path, default=Path("backups"))
    parser.add_argument("--collection", default=os.environ.get("QDRANT_COLLECTION", "rag_chunks"))
    parser.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", "http://127.0.0.1:6333"))
    parser.add_argument("--rclone-destination", default=os.environ.get("BACKUP_RCLONE_DESTINATION"),
                        help="optional rclone remote path; configure it as an rclone crypt remote for encryption")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    try:
        if args.verify:
            print(json.dumps(verify_backup(args.verify), indent=2))
            return 0
        print(json.dumps(create_backup(args.output_dir, args.collection, args.qdrant_url, args.timeout,
                                       args.rclone_destination), indent=2))
        return 0
    except (OSError, RuntimeError, ValueError, URLError, subprocess.TimeoutExpired) as error:
        print(f"Backup failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
