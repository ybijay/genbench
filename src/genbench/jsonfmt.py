"""Readable JSON for files genbench writes: short objects stay on one line, long ones break."""

from __future__ import annotations

import json
from typing import Any

WIDTH = 110


def dumps(value: Any, width: int = WIDTH) -> str:
    return _fmt(value, 0, 0, width) + "\n"


def _fmt(value: Any, indent: int, start: int, width: int) -> str:
    flat = json.dumps(value, ensure_ascii=False)
    if not isinstance(value, (dict, list)) or not value or start + len(flat) <= width:
        return flat
    pad = " " * (indent + 2)
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            head = f"{pad}{json.dumps(key, ensure_ascii=False)}: "
            parts.append(head + _fmt(item, indent + 2, len(head), width))
        return "{\n" + ",\n".join(parts) + "\n" + " " * indent + "}"
    parts = [pad + _fmt(item, indent + 2, len(pad), width) for item in value]
    return "[\n" + ",\n".join(parts) + "\n" + " " * indent + "]"
