"""Ask a model server whether it is up and which models it has."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import httpx

from genbench.config import Endpoint


@dataclass
class ProbeResult:
    endpoint: str
    url: str | None = None
    models: list[str] = field(default_factory=list)
    error: str | None = None
    families: dict[str, str] = field(default_factory=dict)  # model -> family, when the server reports it

    @property
    def ok(self) -> bool:
        return self.url is not None


def probe(endpoint: Endpoint, timeout: float = 3.0) -> ProbeResult:
    """Try the endpoint's URL, then its fallbacks; return the first that answers."""
    headers = {}
    if endpoint.api_key_env and os.environ.get(endpoint.api_key_env):
        headers["Authorization"] = f"Bearer {os.environ[endpoint.api_key_env]}"
    errors = []
    for url in endpoint.urls():
        target = f"{url}/api/tags" if endpoint.kind == "ollama" else f"{url}/models"
        try:
            response = httpx.get(target, headers=headers, timeout=timeout)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            errors.append(f"{url}: {type(exc).__name__}")
            continue
        families: dict[str, str] = {}
        if endpoint.kind == "ollama":
            names = [m.get("name") or m.get("model") for m in data.get("models", [])]
            for m in data.get("models", []):
                family = (m.get("details") or {}).get("family")
                if family and (m.get("name") or m.get("model")):
                    families[m.get("name") or m.get("model")] = str(family)
        else:
            names = [m.get("id") for m in data.get("data", [])]
        return ProbeResult(endpoint.name, url, [n for n in names if n], families=families)
    return ProbeResult(endpoint.name, error="; ".join(errors) or "no URL to try")


def is_embedding_model(name: str, family: str = "") -> bool:
    """Embedding models can't chat, so a benchmark run leaves them out."""
    return "embed" in name.lower() or "bert" in family.lower()


def chat_models(result: ProbeResult) -> list[str]:
    return [name for name in result.models if not is_embedding_model(name, result.families.get(name, ""))]


def has_model(available: list[str], wanted: str) -> bool:
    """Ollama treats 'llama3' as 'llama3:latest'."""
    return wanted in available or (":" not in wanted and f"{wanted}:latest" in available)
