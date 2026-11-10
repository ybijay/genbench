"""Model servers. Ollama's native API in Phase 1; OpenAI-compatible servers arrive in Phase 2."""

from __future__ import annotations

from genbench.backends.base import Attempt, BackendError, Warmup
from genbench.config import Endpoint

__all__ = ["Attempt", "BackendError", "Warmup", "make_backend"]


def make_backend(endpoint: Endpoint):  # -> OllamaBackend
    if endpoint.kind == "ollama":
        from genbench.backends.ollama import OllamaBackend

        return OllamaBackend(endpoint)
    raise BackendError(
        f"endpoint '{endpoint.name}' is kind '{endpoint.kind}'; OpenAI-compatible servers arrive in Phase 2"
    )
