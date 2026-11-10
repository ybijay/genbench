"""Tool calls written as JSON in the answer text, parsed exactly as the original grader did."""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Iterable
from typing import Any

from genbench.pack.checks import ExpectedCall

JSON_FENCE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
JSON_ARRAY = re.compile(r"\[\s*\{.*?\}\s*\]", re.DOTALL)


def calls_from_text(text: str) -> list[Any]:
    """First fenced block holding a JSON list (or {"tool_calls": [...]}), else the first [{...}] in the text."""
    return locate_calls(text)[0]


def locate_calls(text: str) -> tuple[list[Any], bool]:
    """(calls, found): found is False when the text holds no tool-call list at all, not even an empty one."""
    parsed: Any = []
    found = False
    for block in JSON_FENCE.findall(text):
        try:
            obj = json.loads(block.strip())
        except (ValueError, RecursionError):
            continue
        if isinstance(obj, list):
            parsed, found = obj, True
            break
        if isinstance(obj, dict) and "tool_calls" in obj:
            parsed, found = obj["tool_calls"], True
            break
    if not parsed:
        match = JSON_ARRAY.search(text)
        if match:
            with contextlib.suppress(ValueError, RecursionError):
                parsed = json.loads(match.group(0))
                found = True
    if not found:  # the whole text may be a bare list, e.g. "[]" once a fence has been stripped
        with contextlib.suppress(ValueError, RecursionError):
            whole = json.loads(text.strip())
            if isinstance(whole, list):
                parsed, found = whole, True
            elif isinstance(whole, dict) and "tool_calls" in whole:
                parsed, found = whole["tool_calls"], True
    return (parsed if isinstance(parsed, list) else []), found


def normalize_calls(raw: Iterable[Any]) -> list[tuple[str, Any]]:
    """[(name, arguments)] from {"name", "arguments"} or OpenAI-style {"function": {...}} items."""
    calls = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        fn = item.get("function", item)
        if not isinstance(fn, dict):
            continue
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {}
        calls.append((name if isinstance(name, str) else str(name), args))
    return calls


def score_calls(calls: list[tuple[str, Any]], expected: list[ExpectedCall], count_duplicates: bool = True) -> float:
    """Each call earns the points of the first expected call whose name and listed arguments match."""
    total = 0.0
    used: set[int] = set()
    for name, args in calls:
        for i, want in enumerate(expected):
            if name == want.name and isinstance(args, dict) and all(args.get(k) == v for k, v in want.args.items()):
                if count_duplicates or i not in used:
                    total += want.points
                    used.add(i)
                break
    return total
