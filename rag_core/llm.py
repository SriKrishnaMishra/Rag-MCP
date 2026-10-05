"""Local Ollama client for grounded RAG answers."""
from __future__ import annotations

import httpx


class OllamaClient:
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url, self.model = base_url.rstrip("/"), model

    def answer(self, question: str, sources: list[dict[str, object]], model: str | None = None) -> str:
        context = "\n\n".join(f"[Source {index + 1}] {item['content']}" for index, item in enumerate(sources))
        prompt = ("Answer only from the supplied sources. If the sources do not contain the answer, say so. "
                  f"\n\nSources:\n{context}\n\nQuestion: {question}\nAnswer:")
        try:
            response = httpx.post(f"{self.base_url}/api/generate", json={"model": model or self.model,
                                  "prompt": prompt, "stream": False}, timeout=60)
            response.raise_for_status()
            return str(response.json()["response"]).strip()
        except (httpx.HTTPError, KeyError) as error:
            raise RuntimeError("Ollama is unavailable. Start Ollama and pull the configured model.") from error
