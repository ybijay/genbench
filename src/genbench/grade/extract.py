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
    closed: bool = True  # False when a JSON block's closing ``` was missing and it ran to the end of the reply


def _lang_group(langs: tuple[str, ...]) -> str:
    names = [re.escape(lang) for lang in langs if lang]
    if names:
        return "(?:" + "|".join(names) + ")" + ("?" if "" in langs else "")
    return "" if "" in langs else "[^\\n`]*"


@lru_cache(maxsize=64)
def fence_pattern(langs: tuple[str, ...]) -> re.Pattern[str]:
    """```lang\\n ... ``` for the given languages; '' allows an untagged fence; () allows any tag."""
    return re.compile("```" + _lang_group(langs) + r"\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


@lru_cache(maxsize=64)
def opening_fence_pattern(langs: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile("```" + _lang_group(langs) + r"\s*\n", re.IGNORECASE)


def unclosed_block(text: str, langs: tuple[str, ...]) -> str | None:
    """The body of a fence that opens and is never closed: everything after the last opening fence,
    provided no ``` follows it. None when there is no such fence."""
    openers = list(opening_fence_pattern(langs).finditer(text))
    if not openers:
        return None
    rest = text[openers[-1].end():]
    return None if "```" in rest else rest


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
    if not matches and spec.kind == "json_block":
        # A reply that opens a JSON fence and stops without closing it still holds the answer; grade it
        # (the built-in rubrics take a few quality points for the missing fence). Only for JSON: an
        # unclosed code fence is usually a reply cut off mid-file.
        for langs in (primary, spec.fallback_langs):
            body = unclosed_block(cleaned, tuple(langs)) if langs is not None else None
            if body and body.strip():
                block = body.strip()
                return Artifact(block, cleaned, min(1.0, len(block) / max(1, len(cleaned))), True, closed=False)
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
