import hashlib
import struct

import pytest

from mcp_servers.inspector.approved_actions import (
    apply_approved_source_edit,
    preview_source_edit,
    probe_repository_dependencies,
    run_selected_repository_tests,
)


def test_dependency_probe_requires_confirmation(tmp_path) -> None:
    with pytest.raises(ValueError, match="confirm=True"):
        probe_repository_dependencies(tmp_path)


def test_dependency_probe_only_targets_loopback_and_known_service_ports(tmp_path, monkeypatch) -> None:
    compose = tmp_path / "compose.yaml"
    compose.write_text("""services:
  postgres:
    image: postgres:16
    ports: [\"127.0.0.1:15432:5432\", \"0.0.0.0:5433:5432\"]
  qdrant:
    image: qdrant/qdrant
    ports: [\"127.0.0.1:16333:6333\"]
  redis:
    image: redis:7-alpine
    ports: [\"127.0.0.1:16379:6379\"]
  arbitrary:
    image: example/app
    ports: [\"127.0.0.1:19999:9999\"]
""", encoding="utf-8")

    class Response:
        status = 200
        def read(self, _limit):
            return b"ok"

    class HttpConnection:
        def __init__(self, host, port, timeout):
            assert (host, port, timeout) == ("127.0.0.1", 16333, 2)
        def request(self, method, path):
            assert (method, path) == ("GET", "/healthz")
        def getresponse(self):
            return Response()
        def close(self):
            pass

    class Socket:
        def close(self):
            pass

    monkeypatch.setattr("mcp_servers.inspector.approved_actions.http.client.HTTPConnection", HttpConnection)
    monkeypatch.setattr("mcp_servers.inspector.approved_actions._probe_postgres",
                        lambda host, port, timeout: (host == "127.0.0.1" and port == 15432 and timeout == 2,
                                                     "PostgreSQL startup authentication challenge"))
    monkeypatch.setattr("mcp_servers.inspector.approved_actions._probe_redis",
                        lambda host, port, timeout: (host == "127.0.0.1" and port == 16379 and timeout == 2,
                                                     "Redis PING"))
    result = probe_repository_dependencies(tmp_path, confirm=True, timeout_seconds=2)

    assert [(probe["kind"], probe["healthy"]) for probe in result["probes"]] == [
        ("postgresql", True), ("qdrant", True), ("redis", True)]
    assert result["network_scope"] == "loopback only"


def test_postgres_probe_sends_startup_packet_and_accepts_auth_challenge(monkeypatch) -> None:
    from mcp_servers.inspector.approved_actions import _probe_postgres

    class Connection:
        def __init__(self):
            self.response = bytearray(b"R" + struct.pack("!I", 8) + struct.pack("!I", 3))
            self.packet = b""
        def __enter__(self): return self
        def __exit__(self, *_args): return None
        def settimeout(self, _timeout): pass
        def sendall(self, packet): self.packet = packet
        def recv(self, count):
            block = bytes(self.response[:count])
            del self.response[:count]
            return block

    connection = Connection()
    monkeypatch.setattr("mcp_servers.inspector.approved_actions.socket.create_connection",
                        lambda address, timeout: connection)
    healthy, detail = _probe_postgres("127.0.0.1", 5432, 2)
    assert healthy is True
    assert detail == "PostgreSQL startup authentication challenge"
    assert connection.packet.startswith(struct.pack("!II", len(connection.packet), 196608))
    assert b"probe\x00" in connection.packet


def test_redis_probe_uses_read_only_ping_and_handles_noauth(monkeypatch) -> None:
    from mcp_servers.inspector.approved_actions import _probe_redis

    class Connection:
        def __init__(self, response): self.response, self.command = bytearray(response), b""
        def __enter__(self): return self
        def __exit__(self, *_args): return None
        def settimeout(self, _timeout): pass
        def sendall(self, command): self.command = command
        def recv(self, count):
            block = bytes(self.response[:min(3, count)])
            del self.response[:len(block)]
            return block

    connection = Connection(b"+PONG\r\n")
    monkeypatch.setattr("mcp_servers.inspector.approved_actions.socket.create_connection",
                        lambda *_args, **_kwargs: connection)
    assert _probe_redis("127.0.0.1", 6379, 2) == (True, "Redis PING")
    assert connection.command == b"*1\r\n$4\r\nPING\r\n"

    denied = Connection(b"-NOAUTH Authentication required\r\n")
    monkeypatch.setattr("mcp_servers.inspector.approved_actions.socket.create_connection",
                        lambda *_args, **_kwargs: denied)
    assert _probe_redis("127.0.0.1", 6379, 2) == (False, "Redis requires authentication")


def test_source_edit_requires_hash_match_and_explicit_confirmation(tmp_path) -> None:
    source = tmp_path / "service.py"
    source.write_text("VALUE = 'old'\n", encoding="utf-8")
    preview = preview_source_edit(tmp_path, "service.py", "old", "new")

    with pytest.raises(ValueError, match="confirm=True"):
        apply_approved_source_edit(tmp_path, "service.py", preview["sha256"], "old", "new")
    assert source.read_text(encoding="utf-8") == "VALUE = 'old'\n"

    applied = apply_approved_source_edit(tmp_path, "service.py", preview["sha256"], "old", "new", confirm=True)

    assert applied["updated"] is True
    assert source.read_text(encoding="utf-8") == "VALUE = 'new'\n"


def test_source_edit_refuses_stale_hash_and_paths_outside_selected_repo(tmp_path) -> None:
    source = tmp_path / "service.py"
    source.write_text("VALUE = 'old'\n", encoding="utf-8")
    preview = preview_source_edit(tmp_path, "service.py", "old", "new")
    source.write_text("VALUE = 'changed'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after preview"):
        apply_approved_source_edit(tmp_path, "service.py", preview["sha256"], "old", "new", confirm=True)
    with pytest.raises(ValueError, match="inside the selected repository"):
        preview_source_edit(tmp_path, "..\\outside.py", "old", "new")


def test_source_edit_cannot_skip_the_preview_step(tmp_path) -> None:
    source = tmp_path / "service.py"
    source.write_text("VALUE = 'old'\n", encoding="utf-8")
    current_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="no matching diff preview"):
        apply_approved_source_edit(tmp_path, "service.py", current_hash, "old", "new", confirm=True)


def test_repository_test_execution_requires_confirmation(tmp_path) -> None:
    with pytest.raises(ValueError, match="confirm=True"):
        run_selected_repository_tests(tmp_path)


def test_repository_test_runner_executes_detected_pytest_with_confirmation(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths = ['tests']\naddopts = '-s'\n", encoding="utf-8")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_smoke.py").write_text(
        "def test_smoke():\n    print('API_KEY=should-not-escape')\n    assert 2 + 2 == 4\n", encoding="utf-8")

    result = run_selected_repository_tests(tmp_path, confirm=True, timeout_seconds=30)

    assert result["passed"] is True
    assert result["return_code"] == 0
    assert "should-not-escape" not in result["output"]
