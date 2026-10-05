"""Explicitly approved test execution and hash-guarded source edits."""
from __future__ import annotations

import difflib
import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path
import re
import shutil
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import http.client
from typing import Any

import yaml

from mcp_servers.inspector.connectors import inspect_repository_integrations


EDITABLE_SUFFIXES = {".py", ".toml", ".yml", ".yaml", ".json", ".md"}
MAX_EDIT_BYTES = 2_000_000
MAX_REPLACEMENT_BYTES = 64_000
MAX_OUTPUT_CHARS = 8_000
_PENDING_EDITS: OrderedDict[tuple[str, str, str, str, str], None] = OrderedDict()
MAX_PENDING_EDITS = 128


def _recv_exact(connection: socket.socket, count: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < count:
        block = connection.recv(count - len(chunks))
        if not block:
            raise ConnectionError("service closed the connection during its health response")
        chunks.extend(block)
    return bytes(chunks)


def _probe_postgres(host: str, port: int, timeout: int) -> tuple[bool, str]:
    """Send a credential-free PostgreSQL startup packet and validate its first response."""
    parameters = b"user\x00probe\x00database\x00probe\x00client_encoding\x00UTF8\x00\x00"
    packet = struct.pack("!II", len(parameters) + 8, 196608) + parameters
    with socket.create_connection((host, port), timeout=timeout) as connection:
        connection.settimeout(timeout)
        connection.sendall(packet)
        header = _recv_exact(connection, 5)
        message_type, length = header[:1], struct.unpack("!I", header[1:])[0]
        if not 4 <= length <= 65536:
            return False, "invalid_postgresql_response"
        _recv_exact(connection, length - 4)
        if message_type == b"R":
            return True, "PostgreSQL startup authentication challenge"
        if message_type == b"E":
            return True, "PostgreSQL startup error response (server is reachable)"
        return False, "unexpected_postgresql_response"


def _probe_redis(host: str, port: int, timeout: int) -> tuple[bool, str]:
    """Issue Redis PING without credentials or state-changing commands."""
    with socket.create_connection((host, port), timeout=timeout) as connection:
        connection.settimeout(timeout)
        connection.sendall(b"*1\r\n$4\r\nPING\r\n")
        response = bytearray()
        while not response.endswith(b"\r\n") and len(response) < 256:
            block = connection.recv(256 - len(response))
            if not block:
                break
            response.extend(block)
    response = bytes(response)
    if response.startswith(b"+PONG"):
        return True, "Redis PING"
    if response.startswith(b"-NOAUTH"):
        return False, "Redis requires authentication"
    if response.startswith(b"-"):
        return False, "Redis returned an error to PING"
    return False, "unexpected Redis PING response"


def _redact_output(text: str) -> str:
    text = re.sub(r"(?i)(api[_-]?key|password|secret|token|authorization)\s*[:=]\s*([^\s,;]+)",
                  r"\1=[REDACTED]", text)
    return re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[REDACTED]", text)


def _target(root: Path, relative_path: str) -> Path:
    root = root.resolve()
    requested = Path(relative_path)
    if requested.is_absolute():
        raise ValueError("path must be relative to the selected repository")
    candidate = (root / requested).resolve()
    if candidate == root or not candidate.is_relative_to(root):
        raise ValueError("path must remain inside the selected repository")
    if candidate.suffix.lower() not in EDITABLE_SUFFIXES:
        raise ValueError("only text source and project files may be edited")
    return candidate


def _read_text_preserving_newlines(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as source:
        return source.read()


def preview_source_edit(root: Path, relative_path: str, old_text: str, new_text: str) -> dict[str, object]:
    if max(len(old_text.encode("utf-8")), len(new_text.encode("utf-8"))) > MAX_REPLACEMENT_BYTES:
        raise ValueError("edit snippets exceed the size limit")
    candidate = _target(root, relative_path)
    if not candidate.is_file() or candidate.is_symlink():
        raise ValueError("target must be an existing regular file")
    content = _read_text_preserving_newlines(candidate)
    if len(content.encode("utf-8")) > MAX_EDIT_BYTES:
        raise ValueError("target file is larger than the edit limit")
    if not old_text or content.count(old_text) != 1:
        raise ValueError("old_text must match exactly one location in the current file")
    updated = content.replace(old_text, new_text, 1)
    if len(updated.encode("utf-8")) > MAX_EDIT_BYTES:
        raise ValueError("updated file would exceed the size limit")
    diff = "".join(difflib.unified_diff(content.splitlines(keepends=True),
                                       updated.splitlines(keepends=True),
                                       fromfile=relative_path, tofile=relative_path))
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    approval_key = (str(root.resolve()), relative_path, digest, old_text, new_text)
    _PENDING_EDITS[approval_key] = None
    _PENDING_EDITS.move_to_end(approval_key)
    while len(_PENDING_EDITS) > MAX_PENDING_EDITS:
        _PENDING_EDITS.popitem(last=False)
    return {"path": relative_path, "sha256": digest,
            "diff": diff[:MAX_OUTPUT_CHARS], "diff_truncated": len(diff) > MAX_OUTPUT_CHARS,
            "requires_confirmation": True}


def apply_approved_source_edit(root: Path, relative_path: str, expected_sha256: str,
                               old_text: str, new_text: str, confirm: bool = False) -> dict[str, object]:
    if not confirm:
        raise ValueError("Source changes require confirm=True after reviewing the proposed diff")
    if max(len(old_text.encode("utf-8")), len(new_text.encode("utf-8"))) > MAX_REPLACEMENT_BYTES:
        raise ValueError("edit snippets exceed the size limit")
    candidate = _target(root, relative_path)
    approval_key = (str(root.resolve()), relative_path, expected_sha256, old_text, new_text)
    if approval_key not in _PENDING_EDITS:
        raise ValueError("no matching diff preview exists; prepare and review the exact edit first")
    if not candidate.is_file() or candidate.is_symlink():
        _PENDING_EDITS.pop(approval_key, None)
        raise ValueError("target must be an existing regular file")
    content = _read_text_preserving_newlines(candidate)
    current_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if current_hash != expected_sha256:
        _PENDING_EDITS.pop(approval_key, None)
        raise ValueError("target changed after preview; prepare a new diff before applying")
    if len(content.encode("utf-8")) > MAX_EDIT_BYTES or not old_text or content.count(old_text) != 1:
        _PENDING_EDITS.pop(approval_key, None)
        raise ValueError("old_text must match exactly one location in a file within the edit limit")
    updated = content.replace(old_text, new_text, 1)
    if len(updated.encode("utf-8")) > MAX_EDIT_BYTES:
        _PENDING_EDITS.pop(approval_key, None)
        raise ValueError("updated file would exceed the size limit")
    original_mode = stat.S_IMODE(candidate.stat().st_mode)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=candidate.parent, prefix=f".{candidate.name}.codex-",
                                         suffix=".tmp", delete=False) as temporary:
            temporary_path = temporary.name
            temporary.write(updated)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_path, original_mode)
        os.replace(temporary_path, candidate)
        temporary_path = None
        _PENDING_EDITS.pop(approval_key, None)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return {"path": relative_path, "updated": True,
            "sha256": hashlib.sha256(updated.encode("utf-8")).hexdigest()}


def _minimal_process_environment(temp_home: str) -> dict[str, str]:
    keys = ("PATH", "SYSTEMROOT", "WINDIR", "PATHEXT")
    environment = {key: os.environ[key] for key in keys if key in os.environ}
    environment.update({"HOME": temp_home, "TMP": temp_home, "TEMP": temp_home})
    return environment


def run_selected_repository_tests(root: Path, confirm: bool = False,
                                  timeout_seconds: int = 120) -> dict[str, Any]:
    if not confirm:
        raise ValueError("Running repository code requires confirm=True")
    root = root.resolve()
    if not root.is_dir():
        raise ValueError("selected repository does not exist")
    integrations = inspect_repository_integrations(root)["tests"]
    if "pytest" in integrations["frameworks"] or "python-test-runner" in integrations["frameworks"]:
        python = sys.executable
        local_python = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if local_python.is_file():
            python = str(local_python)
        command = [python, "-m", "pytest", "-q"]
    else:
        package_path = root / "package.json"
        package: dict[str, Any] = {}
        if package_path.is_file():
            try:
                package = json.loads(package_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                package = {}
        test_script = package.get("scripts", {}).get("test", "") if isinstance(package, dict) else ""
        if "jest" in test_script.lower():
            command = ["npm", "test", "--", "--runInBand"]
        elif "vitest" in test_script.lower():
            command = ["npm", "test", "--", "--run"]
        else:
            raise ValueError("no supported pytest, Jest, or Vitest test setup was detected")
        if not shutil.which(command[0]):
            raise ValueError(f"required test runner executable is unavailable: {command[0]}")
    timeout_seconds = max(1, min(int(timeout_seconds), 300))
    with tempfile.TemporaryDirectory(prefix="mcp-rag-tests-") as temp_home:
        captured: list[bytes] = []
        captured_size = 0
        captured_lock = threading.Lock()
        output_limit = MAX_OUTPUT_CHARS * 4

        try:
            process = subprocess.Popen(command, cwd=root, env=_minimal_process_environment(temp_home),
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
        except OSError as error:
            raise RuntimeError(f"unable to start the selected test runner: {command[0]}") from error

        def drain_output() -> None:
            nonlocal captured_size
            assert process.stdout is not None
            while chunk := process.stdout.read(4096):
                with captured_lock:
                    captured.append(chunk)
                    captured_size += len(chunk)
                    while captured_size > output_limit and captured:
                        overflow = captured_size - output_limit
                        first = captured[0]
                        if len(first) <= overflow:
                            captured.pop(0)
                            captured_size -= len(first)
                        else:
                            captured[0] = first[overflow:]
                            captured_size -= overflow

        reader = threading.Thread(target=drain_output, daemon=True)
        reader.start()
        timed_out = False
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            return_code = process.wait()
        reader.join(timeout=5)
        if reader.is_alive() and process.stdout:
            process.stdout.close()
            reader.join(timeout=1)
        with captured_lock:
            raw_output = b"".join(captured)
            output_truncated = captured_size >= output_limit
        output = _redact_output(raw_output.decode("utf-8", errors="replace"))[-MAX_OUTPUT_CHARS:]
        result: dict[str, Any] = {"repository": str(root), "command": command,
                                  "return_code": return_code, "passed": return_code == 0 and not timed_out,
                                  "output": output, "output_truncated": output_truncated}
        if timed_out:
            result.update({"timed_out": True, "timeout_seconds": timeout_seconds})
        return result


def inspect_docker_runtime(root: Path, compose_file: str | None = None,
                           timeout_seconds: int = 20) -> dict[str, object]:
    root = root.resolve()
    integrations = inspect_repository_integrations(root)
    compose_files = integrations["services"]["compose_files"]
    if not compose_files:
        raise ValueError("no Docker Compose file was found in the selected repository")
    if compose_file:
        if compose_file not in compose_files:
            raise ValueError("compose_file must match a discovered Compose file")
        selected_files = [compose_file]
    else:
        preferred = next((name for name in ("compose.yaml", "compose.yml", "docker-compose.yaml", "docker-compose.yml")
                          if name in compose_files), None)
        selected_files = [preferred or compose_files[0]]
    docker = shutil.which("docker")
    if not docker:
        raise RuntimeError("Docker CLI is not installed or is not on PATH")
    try:
        command = [docker, "compose"]
        for relative_path in selected_files:
            command.extend(("-f", str(root / relative_path)))
        command.extend(("ps", "--format", "json"))
        completed = subprocess.run(command, cwd=root,
                                   capture_output=True, text=True, timeout=timeout_seconds,
                                   shell=False, check=False)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Docker Compose status check timed out") from error
    if completed.returncode:
        raise RuntimeError(_redact_output(completed.stderr or "Docker Compose status check failed")[-2000:])
    output = completed.stdout.strip()
    if not output:
        services: list[dict[str, object]] = []
    else:
        try:
            parsed = json.loads(output)
            services = parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            try:
                services = [json.loads(line) for line in output.splitlines() if line.strip()]
            except json.JSONDecodeError as error:
                raise RuntimeError("Docker returned unrecognized Compose status output") from error
    return {"repository": str(root), "compose_file": selected_files[0], "services": [
        {key: item.get(key) for key in ("Service", "State", "Status", "Health", "Publishers") if key in item}
        for item in services if isinstance(item, dict)], "read_only": True}


def probe_repository_dependencies(root: Path, compose_file: str | None = None,
                                  confirm: bool = False, timeout_seconds: int = 3) -> dict[str, object]:
    """Probe only loopback-published PostgreSQL/Qdrant/Redis ports declared in Compose."""
    if not confirm:
        raise ValueError("Network probes require confirm=True")
    if not 1 <= timeout_seconds <= 10:
        raise ValueError("timeout_seconds must be between 1 and 10")
    root = root.resolve()
    integrations = inspect_repository_integrations(root)
    compose_files = integrations["services"]["compose_files"]
    if not compose_files:
        raise ValueError("no Docker Compose file was found in the selected repository")
    if compose_file:
        if compose_file not in compose_files:
            raise ValueError("compose_file must match a discovered Compose file")
        selected = compose_file
    else:
        selected = next((name for name in ("compose.yaml", "compose.yml", "docker-compose.yaml", "docker-compose.yml")
                         if name in compose_files), compose_files[0])
    config = yaml.safe_load((root / selected).read_text(encoding="utf-8")) or {}
    results: list[dict[str, object]] = []
    for name, service in (config.get("services", {}) if isinstance(config, dict) else {}).items():
        if not isinstance(service, dict):
            continue
        signature = f"{name} {service.get('image', '')}".lower()
        kind = next((kind for kind, needles in {
            "postgresql": ("postgres", "pgvector"), "qdrant": ("qdrant",), "redis": ("redis",)
        }.items() if any(needle in signature for needle in needles)), None)
        if not kind:
            continue
        for declaration in service.get("ports", []):
            host_ip, host_port, target_port = None, None, None
            if isinstance(declaration, dict):
                host_ip = declaration.get("host_ip")
                host_port, target_port = declaration.get("published"), declaration.get("target")
            elif isinstance(declaration, (str, int)):
                spec = str(declaration).split(":")
                if len(spec) == 2:
                    host_port, target_port = spec
                elif len(spec) == 3:
                    host_ip, host_port, target_port = spec
            if host_ip not in ("127.0.0.1", "::1"):
                continue
            try:
                port = int(str(host_port).split("/")[0])
                target = int(str(target_port).split("/")[0])
            except (TypeError, ValueError):
                continue
            expected = {"postgresql": 5432, "qdrant": 6333, "redis": 6379}[kind]
            if target != expected or not 1 <= port <= 65535:
                continue
            result: dict[str, object] = {"service": str(name), "kind": kind, "host": "127.0.0.1", "port": port}
            try:
                if kind == "qdrant":
                    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout_seconds)
                    connection.request("GET", "/healthz")
                    response = connection.getresponse()
                    response.read(256)
                    connection.close()
                    result.update({"healthy": response.status == 200, "check": "GET /healthz", "http_status": response.status})
                elif kind == "postgresql":
                    healthy, detail = _probe_postgres("127.0.0.1", port, timeout_seconds)
                    result.update({"healthy": healthy, "check": "PostgreSQL startup protocol", "detail": detail})
                else:
                    healthy, detail = _probe_redis("127.0.0.1", port, timeout_seconds)
                    result.update({"healthy": healthy, "check": "Redis PING", "detail": detail})
            except (OSError, TimeoutError, http.client.HTTPException) as error:
                check = {"qdrant": "GET /healthz", "postgresql": "PostgreSQL startup protocol",
                         "redis": "Redis PING"}[kind]
                result.update({"healthy": False, "check": check,
                               "error": type(error).__name__})
            results.append(result)
    return {"repository": str(root), "compose_file": selected, "probes": results,
            "read_only": True, "network_scope": "loopback only"}
