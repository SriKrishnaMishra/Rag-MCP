"""Read-only inventory of service and test integrations in a selected repository."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tomllib

import yaml


COMPOSE_NAMES = ("compose.yml", "compose.yaml", "docker-compose.yml", "docker-compose.yaml")
TEST_CONFIGS = ("pytest.ini", "tox.ini", "noxfile.py", "jest.config.js", "vitest.config.ts",
                "playwright.config.ts", "cypress.config.ts", "Makefile")
IGNORED_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "dist", "build"}
MAX_SCAN_DIRECTORIES = 20_000
MAX_COMPOSE_FILES = 100
MAX_CONFIG_FILE_BYTES = 1_000_000
MAX_CONFIG_BYTES = 10_000_000
MAX_SERVICES = 1_000


def _within(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError, RuntimeError):
        return False


def _compose_service_connectors(root: Path) -> tuple[list[str], list[dict[str, object]], bool]:
    config_files: list[str] = []
    services: list[dict[str, object]] = []
    compose_paths = {root / name for name in COMPOSE_NAMES if (root / name).is_file()}
    truncated = False
    directories = 0
    for directory, child_directories, filenames in os.walk(root):
        directories += 1
        if directories > MAX_SCAN_DIRECTORIES:
            truncated = True
            break
        child_directories[:] = sorted(name for name in child_directories
                                       if not name.startswith(".") and name not in IGNORED_DIRS)
        for filename in sorted(filenames):
            lowered = filename.lower()
            if "compose" not in lowered or not lowered.endswith((".yml", ".yaml")):
                continue
            compose_paths.add(Path(directory) / filename)
            if len(compose_paths) >= MAX_COMPOSE_FILES:
                truncated = True
                break
        if truncated:
            break
    total_bytes = 0
    for path in sorted(compose_paths)[:MAX_COMPOSE_FILES]:
        if not path.is_file() or not _within(root, path):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > MAX_CONFIG_FILE_BYTES or total_bytes + size > MAX_CONFIG_BYTES:
            truncated = True
            continue
        total_bytes += size
        config_files.append(str(path.relative_to(root)))
        try:
            parsed = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, UnicodeError, yaml.YAMLError):
            continue
        definitions = parsed.get("services", {}) if isinstance(parsed, dict) else {}
        if not isinstance(definitions, dict):
            continue
        for name, raw in definitions.items():
            if not isinstance(raw, dict):
                continue
            if len(services) >= MAX_SERVICES:
                truncated = True
                break
            image = str(raw.get("image", ""))
            signature = f"{name} {image}".lower()
            kinds = ["docker"]
            for category, markers in {
                "postgresql": ("postgres", "pgvector"), "qdrant": ("qdrant",),
                "redis": ("redis",), "ollama": ("ollama",),
            }.items():
                if any(marker in signature for marker in markers):
                    kinds.append(category)
            environment = raw.get("environment", {})
            if isinstance(environment, dict):
                env_keys = sorted(str(key) for key in environment)
            elif isinstance(environment, list):
                env_keys = sorted(str(item).split("=", 1)[0] for item in environment)
            else:
                env_keys = []
            ports = []
            for port in raw.get("ports", []):
                if isinstance(port, (str, int)):
                    ports.append(str(port))
                elif isinstance(port, dict) and "target" in port:
                    ports.append(f"{port.get('published', 'dynamic')}:{port['target']}")
            services.append({"name": str(name), "image": image or None, "connectors": kinds,
                             "ports": ports, "environment_keys": env_keys})
    return sorted(set(config_files)), services, truncated


def inspect_repository_integrations(root: Path) -> dict[str, object]:
    """Inventory declared Docker services and test frameworks without executing repository code."""
    root = root.resolve()
    compose_files, services, truncated = _compose_service_connectors(root)
    test_files = [str((root / name).relative_to(root)) for name in TEST_CONFIGS
                  if (root / name).is_file() and _within(root, root / name)]
    frameworks: set[str] = set()
    suggested_commands: set[str] = set()
    pyproject = root / "pyproject.toml"
    if pyproject.is_file() and _within(root, pyproject):
        try:
            project = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            declaration = json.dumps(project).lower()
            if "pytest" in declaration:
                frameworks.add("pytest")
                suggested_commands.add("pytest")
            if "unittest" in declaration:
                frameworks.add("unittest")
        except (OSError, UnicodeError, tomllib.TOMLDecodeError):
            pass
    package_json = root / "package.json"
    if package_json.is_file() and _within(root, package_json):
        try:
            package = json.loads(package_json.read_text(encoding="utf-8"))
            scripts = package.get("scripts", {}) if isinstance(package, dict) else {}
            if isinstance(scripts, dict) and "test" in scripts:
                frameworks.add("npm-test-script")
                suggested_commands.add("npm test")
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
    if any(name in test_files for name in ("jest.config.js", "vitest.config.ts")):
        frameworks.add("javascript-test-runner")
    if "pytest.ini" in test_files or "tox.ini" in test_files:
        frameworks.add("python-test-runner")
        suggested_commands.add("pytest")
    return {"repository": str(root), "mode": "read_only_inventory", "scan_truncated": truncated,
            "services": {"compose_files": compose_files, "entries": services},
            "tests": {"config_files": test_files, "frameworks": sorted(frameworks),
                      "suggested_commands": sorted(suggested_commands),
                      "executed": False}}
