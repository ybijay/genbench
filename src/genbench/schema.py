"""JSON Schema for test packs, generated from the pydantic models so the two never disagree."""

from __future__ import annotations

import json
from typing import Any

from genbench.pack.models import Pack

DIALECT = "https://json-schema.org/draft/2020-12/schema"


def pack_schema() -> dict[str, Any]:
    generated = Pack.model_json_schema(by_alias=True)
    schema: dict[str, Any] = {"$schema": DIALECT, "title": "genbench test pack"}
    schema.update({k: v for k, v in generated.items() if k != "title"})
    # Top-level rubric checks must carry an 'expect' line; nested ones needn't.
    rubric = schema["$defs"]["Rubric"]["properties"]
    for section in ("accuracy", "quality"):
        prop = rubric[section]
        arrays = [prop] if prop.get("type") == "array" else [p for p in prop.get("anyOf", []) if p.get("type") == "array"]
        for array in arrays:
            array["items"] = {"allOf": [array["items"]], "required": ["expect"]}
    return schema


def schema_text() -> str:
    return json.dumps(pack_schema(), indent=2, ensure_ascii=False) + "\n"
