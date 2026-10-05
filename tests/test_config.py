import pytest

from app.config import Settings


def test_authentication_cannot_start_without_any_api_keys(monkeypatch):
    monkeypatch.setenv("TENANT_API_KEYS", "{}")
    with pytest.raises(ValueError, match="requires at least one TENANT_API_KEYS entry"):
        Settings(auth_required=True)


def test_authentication_starts_with_a_configured_api_key(monkeypatch):
    monkeypatch.setenv("TENANT_API_KEYS", '{"example-key":"tenant-a"}')
    settings = Settings(auth_required=True)
    assert settings.tenant_for_key("example-key") == "tenant-a"


def test_persistent_backend_requires_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(ValueError, match="DATABASE_URL is required"):
        Settings(rag_storage_backend="persistent", database_url="")


def test_empty_qdrant_api_key_is_treated_as_unconfigured():
    assert Settings(qdrant_api_key="").qdrant_api_key is None


@pytest.mark.parametrize("raw, expected", [("true", True), ("YES", True), ("1", True),
                                             ("false", False), ("OFF", False), ("0", False)])
def test_boolean_environment_settings_parse_explicit_values(monkeypatch, raw, expected):
    monkeypatch.setenv("AUTH_REQUIRED", raw)
    monkeypatch.setenv("RATE_LIMIT_FAIL_OPEN", raw)
    monkeypatch.setenv("TENANT_API_KEYS", '{"example-key":"tenant-a"}')
    settings = Settings()
    assert settings.auth_required is expected
    assert settings.rate_limit_fail_open is expected


@pytest.mark.parametrize("name", ["AUTH_REQUIRED", "RATE_LIMIT_FAIL_OPEN"])
def test_boolean_environment_settings_reject_typos(monkeypatch, name):
    monkeypatch.setenv(name, "treu")
    with pytest.raises(ValueError, match=f"{name} must be a boolean"):
        Settings()
