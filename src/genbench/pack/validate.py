"""Checks beyond the JSON shape: ids, references, regexes, tools, files and placeholders."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from genbench.config import Config
from genbench.pack.checks import COMPOSITES
from genbench.pack.loader import Catalog, Issue, LoadedTest

# Lookaround, named or numbered backreferences, conditionals: fine in Python's re, missing from Rust's regex.
NOT_PORTABLE = re.compile(r"\(\?<?[=!]|\(\?P=|\\[1-9]|\(\?\(")
PLACEHOLDER = "TODO"
_FLAGS = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL, "x": re.VERBOSE}


def validate_catalog(catalog: Catalog, cfg: Config, *, portable: bool = False) -> list[Issue]:
    issues = list(catalog.issues)

    pack_names: dict[str, str] = {}
    for lp in catalog.packs:
        name = lp.pack.pack
        if name in pack_names:
            issues.append(Issue("error", str(lp.path), f"pack name '{name}' is also used by {pack_names[name]}"))
        else:
            pack_names[name] = str(lp.path)

    keys: dict[str, str] = {}
    for lt in catalog.tests:
        for key in [lt.test.id, *lt.test.aliases]:
            if key in keys:
                issues.append(Issue("error", lt.where, f"id or alias '{key}' is also used by {keys[key]}"))
            else:
                keys[key] = lt.where
        issues.extend(_validate_test(lt, cfg, portable))

    categories = {lt.test.category for lt in catalog.tests}
    for key in cfg.scoring.weights:
        if key.startswith("category.") and key.split(".", 1)[1] not in categories:
            issues.append(
                Issue("warning", "[scoring.weights]", f"'{key}' matches no loaded test, so it adds nothing to Overall")
            )
    return issues


def _validate_test(lt: LoadedTest, cfg: Config, portable: bool) -> list[Issue]:
    test = lt.test
    out: list[Issue] = []

    def error(message: str) -> None:
        out.append(Issue("error", lt.where, message))

    def warning(message: str) -> None:
        out.append(Issue("warning", lt.where, message))

    # Prompt
    prompt: str | None = None
    if test.prompt_file is not None and not (lt.pack.dir / test.prompt_file).is_file():
        error(f"prompt_file not found: {test.prompt_file}")
    else:
        try:
            prompt = lt.prompt_text()
        except (OSError, UnicodeDecodeError) as exc:
            error(f"can't read prompt_file {test.prompt_file}: {exc}")
    if test.context is not None:
        from genbench.context import generate

        try:
            generate(test.context)
        except ValueError as exc:
            error(f"context can't be generated: {exc}")
        if prompt is not None and not re.search(r"\{\{\s*context\s*\}\}", prompt):
            warning("the test generates a context but its prompt has no {{context}} placeholder")
    if prompt is not None:
        if PLACEHOLDER in prompt:
            warning(f"the prompt still contains {PLACEHOLDER}")
        defined = set(test.vars) | ({"context"} if test.context is not None else set())
        for var in sorted(set(re.findall(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}", prompt))):
            if var not in defined:
                warning(f"the prompt uses {{{{{var}}}}} but 'vars' doesn't define it")

    # Rubric
    for section, _ in test.rubric.sections():
        if test.rubric.max_score(section) <= 0:
            error(f"rubric.{section} can't score above 0; give its checks points")

    seen_ids: list[str] = []
    for section, path, check, _depth, parent in test.walk_checks():
        kind = check.check
        if check.id:
            if check.id in seen_ids:
                error(f"{path}: check id '{check.id}' is used twice")
            seen_ids.append(check.id)
        if parent in COMPOSITES and check.points:
            warning(f"{path}: points on a check inside {parent} are ignored; the outer check's points count")
        if parent == "fallback" and check.points:
            warning(f"{path}: a fallback earns the validator's points when it passes; its own points are ignored")

        if kind in ("regex", "not_regex"):
            _check_pattern(check.pattern, check.flags, check.engine, path, portable, error, warning)
        elif kind == "numeric_answer" and check.pattern or kind == "choice_answer":
            _check_pattern(check.pattern, "", "re", path, portable, error, warning, need_group=True)
        elif kind == "passed":
            earlier = seen_ids[:-1] if check.id else seen_ids
            if check.ref not in earlier:
                known = ", ".join(earlier) or "none"
                error(f"{path}: ref '{check.ref}' doesn't name an earlier check (earlier ids: {known})")
        elif kind in ("validator", "validator_findings"):
            _check_tool(check.tool, path, cfg, error, warning)
        elif kind == "external":
            exe = check.cmd[0]
            if not shutil.which(exe) and not (lt.pack.dir / exe).exists():
                warning(f"{path}: '{exe}' isn't installed here, so this check will error at run time")
        elif kind == "json_schema" and check.schema_file:
            if not (lt.pack.dir / check.schema_file).is_file():
                error(f"{path}: schema_file not found: {check.schema_file}")
        elif kind in ("table_row", "no_table_row"):
            for pattern in check.match:
                _check_pattern(pattern, "i", "re", path, portable, error, warning)
        elif kind == "table_rows":
            for pattern in check.all_match:
                _check_pattern(pattern, "i", "re", path, portable, error, warning)
            if check.max_rows is not None and check.max_rows < check.min_rows:
                error(f"{path}: max_rows is below min_rows")
        elif kind == "scaled_score" and section == "accuracy":
            error(f"{path}: scaled_score reads the accuracy score, so it belongs in rubric.quality")

        if kind in ("contains", "not_contains") and check.value.strip() == PLACEHOLDER:
            warning(f"{path}: placeholder value '{PLACEHOLDER}'; replace it with what the answer should contain")

    if test.extract.kind == "code_block" and not test.extract.langs:
        warning("extract.kind is code_block but 'langs' is empty, so any fenced block counts")
    return out


def _check_pattern(
    pattern: str, flags: str, engine: str, path: str, portable: bool, error: Any, warning: Any, need_group: bool = False
) -> None:
    module: Any = re
    if engine == "regex":
        try:
            import regex as module  # type: ignore[import-not-found, no-redef]
        except ImportError:
            warning(f"{path}: engine 'regex' needs the optional regex package (pip install regex)")
            return
    value = 0
    for flag in flags:
        value |= _FLAGS[flag]
    try:
        compiled = module.compile(pattern, value)
    except Exception as exc:  # re.error, regex.error
        error(f"{path}: pattern doesn't compile: {exc}")
        return
    if need_group and compiled.groups < 1:
        error(f"{path}: pattern needs a capture group around the answer")
    if portable and (engine == "regex" or NOT_PORTABLE.search(pattern)):
        warning(f"{path}: uses features Rust's default regex engine lacks (lookaround, backreferences)")


def _check_tool(name: str, path: str, cfg: Config, error: Any, warning: Any) -> None:
    tool = cfg.validators.tools.get(name)
    if tool is None:
        defined = ", ".join(sorted(cfg.validators.tools)) or "none"
        error(f"{path}: validator '{name}' isn't defined in [validators] (defined: {defined})")
        return
    if tool.cmd:
        exe = tool.cmd[0]
        if not shutil.which(exe) and not Path(exe).is_file():
            policy = {
                "fallback": "its fallback check is used instead",
                "skip": "its points are left out",
                "zero": "it scores 0",
            }[cfg.validators.on_missing]
            warning(f"{path}: '{exe}' for validator '{name}' isn't installed here; {policy} (on_missing)")
