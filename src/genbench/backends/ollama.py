"""Ollama's native API: warm-up and VRAM inspection, streamed chat with timing, unload."""

from __future__ import annotations

import contextlib
import json
import re
import time
from typing import Any

import httpx

from genbench.backends.base import (
    Attempt,
    BackendError,
    Warmup,
    detect_model_type,
    split_options,
    think_value,
    timing_metrics,
)
from genbench.config import Endpoint


def context_label(tokens: int) -> str:
    return f"{tokens // 1024}k" if tokens >= 1024 else str(tokens)


def _same_model(listed: str, wanted: str) -> bool:
    def norm(name: str) -> str:
        name = name.strip().lower()
        return name if ":" in name else name + ":latest"

    return bool(listed) and norm(listed) == norm(wanted)


class OllamaBackend:
    kind = "ollama"

    def __init__(self, endpoint: Endpoint, client: httpx.Client | None = None):
        self.endpoint = endpoint
        self.client = client or httpx.Client(timeout=httpx.Timeout(endpoint.timeout_s, connect=10.0))
        self.base_url: str | None = None
        self.version: str | None = None
        self._shown: dict[str, dict[str, Any]] = {}

    def close(self) -> None:
        self.client.close()

    # ------------------------------------------------------------------ connection

    def connect(self) -> str:
        """Use the first of url and fallback_urls that answers /api/tags."""
        errors = []
        for url in self.endpoint.urls():
            try:
                self.client.get(f"{url}/api/tags", timeout=5.0).raise_for_status()
            except httpx.HTTPError as exc:
                errors.append(f"{url} ({type(exc).__name__})")
                continue
            self.base_url = url
            try:
                self.version = self.client.get(f"{url}/api/version", timeout=5.0).json().get("version")
            except (httpx.HTTPError, ValueError, AttributeError):
                self.version = None
            return url
        raise BackendError(f"can't reach endpoint '{self.endpoint.name}': " + ", ".join(errors))

    def _url(self, path: str) -> str:
        if self.base_url is None:
            self.connect()
        return f"{self.base_url}{path}"

    # ------------------------------------------------------------------ model facts

    def show(self, model: str) -> dict[str, Any]:
        """What the server knows about a model (/api/show), asked once per model; {} when it can't say."""
        if model not in self._shown:
            data: Any = {}
            try:
                response = self.client.post(self._url("/api/show"), json={"model": model, "name": model}, timeout=30.0)
                if response.status_code == 200:
                    data = response.json()
            except (httpx.HTTPError, ValueError):
                data = {}
            self._shown[model] = data if isinstance(data, dict) else {}
        return self._shown[model]

    def model_type(self, model: str) -> str | None:
        return detect_model_type(self.show(model), model)

    # ------------------------------------------------------------------ warm-up

    def warmup(self, model: str, options: dict[str, Any], thinking: str | None = None) -> Warmup:
        """Load the model with a 1-token chat, then read context length, VRAM and GPU share.

        The chat carries the thinking setting, so a value the server refuses shows up here, not test by test."""
        info = Warmup()
        data = self.show(model)
        info.details = data.get("details") or {}
        info.model_type = detect_model_type(data, model)
        match = re.search(r"num_ctx\s+(\d+)", data.get("parameters") or "")
        if match:
            info.context = context_label(int(match.group(1)))
            info.context_tokens = int(match.group(1))

        opts, keep_alive = split_options(options)
        if "num_ctx" in opts:
            info.context = context_label(int(opts["num_ctx"]))
            info.context_tokens = int(opts["num_ctx"])
        body = {
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
            "keep_alive": keep_alive if keep_alive is not None else "15m",
            "options": {**opts, "num_predict": 1, "temperature": 0.0},
        }
        if think_value(thinking) is not None:
            body["think"] = think_value(thinking)
        started = time.perf_counter()
        try:
            response = self.client.post(self._url("/api/chat"), json=body)
            data = response.json() if response.content else {}
            if response.status_code >= 400 or "error" in data:
                raise BackendError(str(data.get("error") or f"HTTP {response.status_code}"))
        except (httpx.HTTPError, ValueError, BackendError) as exc:
            info.error = str(exc)
            info.label = f"Warmup error: {str(exc)[:40]}"
            return info
        load_ns = data.get("load_duration", 0) or 0
        info.cold_load_s = round(load_ns / 1e9, 2) if load_ns else round(time.perf_counter() - started, 2)

        try:
            running = self.client.get(self._url("/api/ps"), timeout=5.0).json().get("models", [])
        except (httpx.HTTPError, ValueError, AttributeError):
            running = []
        for item in running:
            if not (_same_model(item.get("name", ""), model) or _same_model(item.get("model", ""), model)):
                info.others.append(item.get("name") or item.get("model") or "?")
        for item in running:
            if _same_model(item.get("name", ""), model) or _same_model(item.get("model", ""), model):
                size = item.get("size", 0) or 0
                vram = item.get("size_vram", 0) or 0
                if (item.get("context_length") or 0) >= 1024:
                    info.context = context_label(item["context_length"])
                    info.context_tokens = int(item["context_length"])
                info.total_gb = round(size / 1024**3, 1)
                info.vram_gb = round(vram / 1024**3, 1)
                info.gpu_pct = round(vram / size * 100.0, 0) if size > 0 else 100.0
                info.label = f"{info.vram_gb:.1f}GB ({int(info.gpu_pct)}% GPU @ {info.context})"
                break
        return info

    # ------------------------------------------------------------------ chat

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
        *,
        tools: list[dict[str, Any]] | None = None,
        fmt: Any = None,
        timeout_s: float | None = None,
        thinking: str | None = None,
    ) -> Attempt:
        opts, keep_alive = split_options(options)
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": True, "options": opts}
        if think_value(thinking) is not None:
            payload["think"] = think_value(thinking)
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if tools:
            payload["tools"] = tools
        if fmt is not None:
            payload["format"] = fmt
        timeout = httpx.Timeout(timeout_s or self.endpoint.timeout_s, connect=10.0)

        content: list[str] = []
        thinking: list[str] = []
        calls: list[Any] = []
        final: dict[str, Any] = {}
        first_any = first_content = first_thinking = None
        started = time.perf_counter()
        try:
            with self.client.stream("POST", self._url("/api/chat"), json=payload, timeout=timeout) as response:
                if response.status_code >= 400:
                    body = response.read().decode("utf-8", "replace")
                    with contextlib.suppress(ValueError, AttributeError):
                        body = json.loads(body).get("error", body)
                    raise BackendError(f"HTTP {response.status_code}: {str(body)[:300]}")
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    packet = json.loads(line)
                    if packet.get("error"):
                        raise BackendError(str(packet["error"]))
                    message = packet.get("message") or {}
                    piece = message.get("content") or ""
                    thought = message.get("thinking") or ""
                    now = time.perf_counter()
                    if (piece or thought or message.get("tool_calls")) and first_any is None:
                        first_any = now
                    if piece:
                        content.append(piece)
                        first_content = first_content or now
                    if thought:
                        thinking.append(thought)
                        first_thinking = first_thinking or now
                    if message.get("tool_calls"):
                        calls.extend(message["tool_calls"])
                    if packet.get("done"):
                        final = packet
                        break
        except httpx.HTTPError as exc:
            raise BackendError(f"{type(exc).__name__}: {exc}") from None
        except ValueError as exc:
            raise BackendError(f"bad response from server: {exc}") from None
        ended = time.perf_counter()

        text = "".join(content)
        metrics = timing_metrics(started, first_any, ended, final, text)
        return Attempt(
            content=text,
            thinking="".join(thinking),
            tool_calls=calls,
            first_content_s=round(first_content - started, 2) if first_content else None,
            first_thinking_s=round(first_thinking - started, 2) if first_thinking else None,
            done_reason=str(final.get("done_reason") or ""),
            **metrics,
        )

    def unload(self, model: str) -> None:
        with contextlib.suppress(httpx.HTTPError):
            self.client.post(self._url("/api/generate"), json={"model": model, "keep_alive": 0}, timeout=30.0)
