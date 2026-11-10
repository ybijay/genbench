"""What every backend returns, and the timing math shared with the original script."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

MODEL_TYPE_LABELS = {"moe": "MoE", "dense": "Dense"}
ACTIVE_PARAMS = re.compile(r"(^|[-_:.])a\d+(\.\d+)?b($|[-_:.])")  # "35b-a3b": 35B parameters, 3B active per token


def detect_model_type(shown: Mapping[str, Any], name: str) -> str | None:
    """'moe' or 'dense' from what the server reports about a model; None when it reports nothing.

    A mixture-of-experts model runs only a few of its experts for each token, which is why it generates much
    faster than a dense model of the same size. Signs of one, strongest first: an expert count in the model's
    metadata, 'moe' in its family name, or an active-parameter tag such as 'a3b' in its name.
    """
    info = shown.get("model_info") or {}
    for key, value in info.items():
        if str(key).endswith(".expert_count") and isinstance(value, (int, float)) and value > 0:
            return "moe"
    details = shown.get("details") or {}
    families = [details.get("family") or "", *(details.get("families") or [])]
    if any("moe" in str(family).lower() for family in families):
        return "moe"
    if ACTIVE_PARAMS.search(name.lower()):
        return "moe"
    return "dense" if (info or details) else None


class BackendError(Exception):
    """The server couldn't be reached or refused the request; the message is ready to show."""


@dataclass
class Warmup:
    cold_load_s: float = 0.0
    vram_gb: float = 0.0
    total_gb: float = 0.0
    gpu_pct: float | None = None
    context: str = "default"
    context_tokens: int | None = None  # the model's context window, when the server reports it
    model_type: str | None = None  # "moe" or "dense", when known
    label: str = "N/A"
    details: dict[str, Any] = field(default_factory=dict)
    others: list[str] = field(default_factory=list)  # other models loaded on the server at the same time
    error: str | None = None


@dataclass
class Attempt:
    content: str = ""
    thinking: str = ""
    tool_calls: list[Any] = field(default_factory=list)
    total_s: float = 0.0
    wall_ttft_s: float = 0.0
    warm_ttft_s: float = 0.0
    first_content_s: float | None = None
    first_thinking_s: float | None = None
    tokens_per_sec: float = 0.0
    prompt_eval_tps: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    load_s: float = 0.0
    done_reason: str = ""  # why the server stopped: "stop", or "length" when it ran out of room


def think_value(thinking: str | None) -> bool | str | None:
    """Ollama's 'think' request field: off -> false, on -> true, a level as itself; None sends nothing."""
    if thinking is None:
        return None
    return {"off": False, "on": True}.get(thinking, thinking)


def split_options(options: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
    """keep_alive is a request field, not a model option; max_tokens means num_predict to Ollama."""
    opts = dict(options)
    keep_alive = opts.pop("keep_alive", None)
    if "max_tokens" in opts:
        opts.setdefault("num_predict", opts.pop("max_tokens"))
    return opts, keep_alive


def timing_metrics(t_start: float, t_first: float | None, t_end: float, final: Mapping[str, Any], content: str) -> dict[str, Any]:
    """Wall and warm TTFT plus generation and prompt-eval speed, computed exactly as the original script did."""
    total_s = round(t_end - t_start, 2)
    wall_ttft_s = round(t_first - t_start, 2) if t_first is not None else total_s
    load_ns = final.get("load_duration", 0) or 0
    prompt_count = final.get("prompt_eval_count", 0) or 0
    prompt_ns = final.get("prompt_eval_duration", 0) or 0
    eval_count = final.get("eval_count", 0) or 0
    eval_ns = final.get("eval_duration", 0) or 0

    load_s = round(load_ns / 1e9, 2)
    warm_ttft_s = round(max(0.05, wall_ttft_s - load_s), 2)
    if prompt_ns > 0 and warm_ttft_s > (prompt_ns / 1e9) * 3:
        warm_ttft_s = round(prompt_ns / 1e9, 2)
    if eval_ns > 0 and eval_count > 0:
        tokens_per_sec = round(eval_count / (eval_ns / 1e9), 2)
    else:
        gen_time = max(0.05, total_s - wall_ttft_s)
        estimated = eval_count or max(1, len(content) // 4)
        tokens_per_sec = round(estimated / gen_time, 2)
    prompt_eval_tps = round(prompt_count / (prompt_ns / 1e9), 1) if prompt_ns > 0 else 0.0
    return {
        "total_s": total_s,
        "wall_ttft_s": wall_ttft_s,
        "warm_ttft_s": warm_ttft_s,
        "tokens_per_sec": tokens_per_sec,
        "prompt_eval_tps": prompt_eval_tps,
        "prompt_tokens": int(prompt_count),
        "completion_tokens": int(eval_count),
        "load_s": load_s,
    }
