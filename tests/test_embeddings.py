import httpx
import pytest

from rag_core.embeddings import OllamaEmbedder
from rag_core.errors import TransientDependencyError


def test_ollama_connection_errors_are_classified_as_transient(monkeypatch):
    monkeypatch.setattr("rag_core.embeddings.httpx.post", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        httpx.ConnectError("connection refused")))
    with pytest.raises(TransientDependencyError, match="temporarily unreachable"):
        OllamaEmbedder("http://localhost:11434", "embed-model").embed("text")


@pytest.mark.parametrize("status", [429, 503])
def test_ollama_rate_limit_and_server_errors_are_transient(monkeypatch, status):
    response = httpx.Response(status, request=httpx.Request("POST", "http://localhost/api/embed"))
    monkeypatch.setattr("rag_core.embeddings.httpx.post", lambda *_args, **_kwargs: response)
    with pytest.raises(TransientDependencyError, match="temporary error"):
        OllamaEmbedder("http://localhost:11434", "embed-model").embed("text")


def test_ollama_model_not_found_is_not_transient(monkeypatch):
    response = httpx.Response(404, request=httpx.Request("POST", "http://localhost/api/embed"))
    monkeypatch.setattr("rag_core.embeddings.httpx.post", lambda *_args, **_kwargs: response)
    with pytest.raises(RuntimeError, match="verify the configured model") as error:
        OllamaEmbedder("http://localhost:11434", "missing-model").embed("text")
    assert not isinstance(error.value, TransientDependencyError)
