"""Turn a model response into the artifact a test grades, following the original script's rules."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from genbench.pack.models import Extract

THINK_TAGS = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str | None) -> str:
    """Remove inline <think>...</think> blocks, then surrounding whitespace."""
    if not text:
        return ""
    return THINK_TAGS.sub("", text).strip()


@dataclass(frozen=True)
class Artifact:
    text: str  # what checks read by default
    response: str  # the whole response, thinking removed (unless strip_thinking is off)
    ratio: float  # share of the response that is the artifact; feeds code_ratio
    found_block: bool  # a fenced block was found


@lru_cache(maxsize=64)
def fence_pattern(langs: tuple[str, ...]) -> re.Pattern[str]:
    """```lang\\n ... ``` for the given languages; '' allows an untagged fence; () allows any tag."""
    names = [re.escape(lang) for lang in langs if lang]
    if names:
        group = "(?:" + "|".join(names) + ")" + ("?" if "" in langs else "")
    elif "" in langs:
        group = ""
    else:
        group = "[^\\n`]*"
    return re.compile("```" + group + r"\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract(response: str | None, spec: Extract) -> Artifact:
    cleaned = strip_thinking(response) if spec.strip_thinking else (response or "").strip()
    if spec.kind in ("whole", "tool_calls"):
        return Artifact(cleaned, cleaned, 1.0, False)
    if not cleaned:
        return Artifact("", cleaned, 0.0, False)
    if spec.kind == "final_answer":
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        return Artifact(lines[-1] if lines else "", cleaned, 1.0, False)

    primary = spec.langs if spec.kind == "code_block" else (spec.langs or ["json", ""])
    matches = fence_pattern(tuple(primary)).findall(cleaned)
    if not matches and spec.fallback_langs is not None:
        matches = fence_pattern(tuple(spec.fallback_langs)).findall(cleaned)
    if not matches:
        if spec.on_no_block == "whole_response":
            return Artifact(cleaned, cleaned, 1.0, False)
        return Artifact("", cleaned, 0.0, False)
    if spec.pick == "first":
        block = matches[0]
    elif spec.pick == "last":
        block = matches[-1]
    else:
        block = max(matches, key=len)
    block = block.strip()
    return Artifact(block, cleaned, min(1.0, len(block) / max(1, len(cleaned))), True)
