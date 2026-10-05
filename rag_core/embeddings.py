"""Embedding provider contracts. Replace HashEmbedder with FastEmbed/SentenceTransformers in production."""
from __future__ import annotations

from hashlib import sha256
from math import sqrt

import httpx

from rag_core.errors import TransientDependencyError


class HashEmbedder:
    """Deterministic no-download development embedder, useful only for wiring and tests."""
    model_name = "hash-embedding-dev-v1"

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in text.lower().split():
            digest = sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0 if digest[4] % 2 else -1.0
        magnitude = sqrt(sum(value * value for value in vector))
        return [value / magnitude for value in vector] if magnitude else vector


class OllamaEmbedder:
    """Production-capable local embedder backed by an Ollama embedding model."""
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url, self.model = base_url.rstrip("/"), model
        self.model_name = f"ollama:{model}"

    def embed(self, text: str) -> list[float]:
        try:
            response = httpx.post(f"{self.base_url}/api/embed", json={"model": self.model, "input": text}, timeout=60)
            response.raise_for_status()
            return list(response.json()["embeddings"][0])
        except httpx.TransportError as error:
            raise TransientDependencyError(
                "Ollama embedding service is temporarily unreachable; retry the request.") from error
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 429 or error.response.status_code >= 500:
                raise TransientDependencyError(
                    "Ollama embedding service returned a temporary error; retry the request.") from error
            raise RuntimeError("Ollama embedding request failed; verify the configured model.") from error
        except (httpx.HTTPError, KeyError, IndexError) as error:
            raise RuntimeError("Ollama embedding service is unavailable; pull the configured embedding model.") from error
