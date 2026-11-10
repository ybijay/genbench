"""Starter files: `genbench init` (config and folders) and `genbench new-test` (a test to fill in)."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from genbench.config import (
    CONFIG_FILENAME,
    Endpoint,
    ModelEntry,
    default_bash_cmd,
    default_endpoint,
    load_config,
    read_models_file,
)
from genbench.jsonfmt import dumps
from genbench.pack.models import Pack
from genbench.schema import schema_text

SCHEMA_RELPATH = Path("schemas") / "testpack.schema.json"


@dataclass
class InitResult:
    config_path: Path
    notes: list[str] = field(default_factory=list)


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        return "{ " + ", ".join(f"{toml_key(k)} = {toml_value(v)}" for k, v in value.items()) + " }"
    raise TypeError(f"can't write {type(value).__name__} as TOML")


def toml_key(key: str) -> str:
    return key if re.fullmatch(r"[A-Za-z0-9_-]+", key) else json.dumps(key, ensure_ascii=False)


def render_models(models: Sequence[ModelEntry], first_endpoint: str) -> list[str]:
    """models = [...]: a plain name per model, or { name, endpoint } for one on another server."""
    items = []
    for model in models:
        if (model.endpoint or first_endpoint) == first_endpoint:
            items.append(toml_value(model.name))
        else:
            items.append(toml_value({"name": model.name, "endpoint": model.endpoint}))
    if len(items) <= 2 and sum(len(item) for item in items) < 80:
        return [f"models = [{', '.join(items)}]"]
    return ["models = [", *[f"  {item}," for item in items], "]"]


def render_minimal_config(endpoints: Sequence[Endpoint], models: Sequence[ModelEntry], write_endpoints: bool) -> str:
    """Only what differs from the defaults: the models, and the servers when they aren't this machine."""
    out = [
        "# genbench configuration. Only what differs from the defaults needs to be here.",
        "# 'genbench init --full' writes every option, and README.md explains them.",
        "",
    ]
    if models:
        out += [*render_models(models, endpoints[0].name), ""]
    else:
        out += [
            "# No models are listed, so 'genbench run' offers the ones installed on the server.",
            "# To fix the list, uncomment this line and edit it:",
            '# models = ["llama3.2:3b"]',
            "",
        ]
    if write_endpoints:
        defaults = Endpoint(name="x", url="x")
        for ep in endpoints:
            out += ["[[endpoints]]", f"name = {toml_value(ep.name)}"]
            if ep.kind != defaults.kind:
                out.append(f"kind = {toml_value(ep.kind)}")
            out.append(f"url = {toml_value(ep.url)}")
            if ep.fallback_urls:
                out.append(f"fallback_urls = {toml_value(ep.fallback_urls)}")
            if ep.timeout_s != defaults.timeout_s:
                out.append(f"timeout_s = {ep.timeout_s:g}")
            if ep.api_key_env:
                out.append(f"api_key_env = {toml_value(ep.api_key_env)}")
            out.append("")
    return "\n".join(out)


def render_config(run_name: str, endpoints: Sequence[Endpoint], models: Sequence[ModelEntry]) -> str:
    """Every section spelled out, for people who want to see all the options (genbench init --full)."""
    out = [
        "# genbench configuration, with every section spelled out. All of it is optional: delete what you",
        "# don't change. Precedence: built-in defaults < this file < GENBENCH_SECTION__KEY env vars < --set flags.",
        "",
        "[run]",
        f"name = {toml_value(run_name)}",
        'output_dir = "runs"',
        "repeats = 1",
        "",
    ]
    for ep in endpoints:
        out += ["[[endpoints]]", f"name = {toml_value(ep.name)}", f"kind = {toml_value(ep.kind)}", f"url = {toml_value(ep.url)}"]
        if ep.fallback_urls:
            out.append(f"fallback_urls = {toml_value(ep.fallback_urls)}")
        out += [f"timeout_s = {ep.timeout_s:g}", ""]
    if models:
        for model in models:
            endpoint = model.endpoint or endpoints[0].name
            out += ["[[models]]", f"name = {toml_value(model.name)}", f"endpoint = {toml_value(endpoint)}", ""]
    else:
        out += [
            "# No models yet. Add one block per model:",
            "# [[models]]",
            '# name = "llama3.2:3b"',
            f"# endpoint = {toml_value(endpoints[0].name)}",
            "",
        ]
    out += [
        "[defaults.options]",
        "temperature = 0.0",
        'keep_alive = "15m"',
        "",
        "[packs]",
        "builtin = true    # the tests that ship with genbench",
        'dirs = ["packs"]  # your own tests: packs/<name>/pack.json',
        "",
        "[validators]",
        'on_missing = "fallback"',
        'python = { builtin = "ast" }',
        f"bash = {{ cmd = {toml_value(default_bash_cmd())} }}",
        "",
        "[scoring]",
        "pass_mark = { accuracy = 70 }",
        "",
        "[report]",
        'formats = ["markdown", "csv", "jsonl"]',
        "",
    ]
    return "\n".join(out)


def endpoints_for(hosts: Sequence[str], kind: str = "ollama") -> list[Endpoint]:
    if not hosts:
        return [default_endpoint()]
    endpoints: list[Endpoint] = []
    taken: set[str] = set()
    for host in hosts:
        endpoint = Endpoint(name="server", kind=kind, url=host)  # type: ignore[arg-type]
        base = re.sub(r"[^A-Za-z0-9_.-]", "-", urlsplit(endpoint.url).hostname or "server").strip("-.") or "server"
        name, n = base, 2
        while name in taken:
            name, n = f"{base}-{n}", n + 1
        taken.add(name)
        endpoints.append(endpoint.model_copy(update={"name": name}))
    return endpoints


def init_project(
    directory: str | Path,
    *,
    hosts: Sequence[str] = (),
    kind: str = "ollama",
    import_models: str | Path | None = None,
    probe: bool = True,
    force: bool = False,
    full: bool = False,
) -> InitResult:
    directory = Path(directory).expanduser().resolve()
    config_path = directory / CONFIG_FILENAME
    if config_path.exists() and not force:
        raise FileExistsError(str(config_path))
    endpoints = endpoints_for(hosts, kind)
    result = InitResult(config_path)

    found: dict[str, list[str]] = {}
    if probe:
        from genbench.backends.probe import probe as probe_endpoint

        for ep in endpoints:
            answer = probe_endpoint(ep)
            if answer.ok:
                found[ep.name] = answer.models
                result.notes.append(f"{ep.name}: reached {answer.url}, {len(answer.models)} model(s) found.")
            else:
                result.notes.append(f"{ep.name}: not reachable ({answer.error}); add its models by hand.")

    if import_models is not None:
        names = read_models_file(import_models)
        models = [ModelEntry(name=n, endpoint=endpoints[0].name) for n in names]
        result.notes.append(f"Imported {len(names)} model(s) from {import_models}.")
    else:
        models = [ModelEntry(name=n, endpoint=ep) for ep, names in found.items() for n in names]

    directory.mkdir(parents=True, exist_ok=True)
    if full:
        text = render_config(directory.name or "local", endpoints, models)
    else:
        text = render_minimal_config(endpoints, models, write_endpoints=bool(hosts))
    config_path.write_text(text, encoding="utf-8")
    (directory / "packs").mkdir(exist_ok=True)
    schema_path = directory / SCHEMA_RELPATH
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(schema_text(), encoding="utf-8")
    load_config(config_path)  # what we wrote must load
    return result


def add_test(
    pack_dir: str | Path,
    *,
    test_id: str,
    name: str,
    description: str,
    goal: str,
    category: str,
    expectations: Sequence[str],
    prompt: str | None = None,
    schema_file: Path | None = None,
) -> Path:
    """Append a test to pack_dir/pack.json (created if missing) with one TODO check per expectation."""
    pack_dir = Path(pack_dir)
    pack_file = pack_dir / "pack.json"
    expectations = [e.strip() for e in expectations if e.strip()]
    if not expectations:
        raise ValueError("add at least one expectation")
    if pack_file.is_file():
        data = json.loads(pack_file.read_text(encoding="utf-8-sig"))
    else:
        data = {}
        if schema_file is not None:
            data["$schema"] = _relative_ref(schema_file, pack_dir)
        pack_name = re.sub(r"[^a-z0-9_-]+", "-", pack_dir.name.lower()).strip("-_") or "my-tests"
        data.update({"pack": pack_name, "version": "0.1.0", "description": "", "tests": []})
    tests = data.setdefault("tests", [])
    if any(isinstance(t, dict) and t.get("id") == test_id for t in tests):
        raise ValueError(f"test '{test_id}' already exists in {pack_file}")
    share = round(100 / len(expectations), 2)
    tests.append(
        {
            "id": test_id,
            "name": name,
            "description": description,
            "goal": goal,
            "category": category,
            "difficulty": "medium",
            "suites": ["standard", "full"],
            "prompt": prompt or f"{'TODO'}: write the exact prompt the model receives.",
            "extract": {"kind": "whole"},
            "rubric": {
                "accuracy": [{"expect": e, "check": "contains", "value": "TODO", "points": share} for e in expectations]
            },
        }
    )
    Pack.model_validate(data)  # refuse to write a pack that won't load
    pack_dir.mkdir(parents=True, exist_ok=True)
    pack_file.write_text(dumps(data), encoding="utf-8")
    return pack_file


def _relative_ref(target: Path, start: Path) -> str:
    try:
        return Path(os.path.relpath(target.resolve(), start.resolve())).as_posix()
    except ValueError:  # different drives on Windows
        return target.resolve().as_uri()
