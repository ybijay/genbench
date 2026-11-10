"""Find and load test packs: the built-in ones plus every pack.json under [packs].dirs."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from genbench.config import Config, PackSettings, clean_message, format_loc
from genbench.pack.models import BenchTest, Pack

BUILTIN_DIR = Path(__file__).resolve().parent.parent / "builtin_packs"
PACK_FILENAME = "pack.json"


@dataclass(frozen=True)
class Issue:
    level: Literal["error", "warning"]
    where: str
    message: str


@dataclass
class LoadedPack:
    pack: Pack
    path: Path
    builtin: bool = False

    @property
    def dir(self) -> Path:
        return self.path.parent

    @property
    def label(self) -> str:
        return f"{self.pack.pack} {self.pack.version}" + (" (built in)" if self.builtin else "")


@dataclass
class LoadedTest:
    test: BenchTest
    pack: LoadedPack

    @property
    def where(self) -> str:
        return f"{self.pack.path} [{self.test.id}]"

    def prompt_text(self) -> str | None:
        return self.test.prompt_text(self.pack.dir)


@dataclass
class Catalog:
    packs: list[LoadedPack] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    @property
    def tests(self) -> list[LoadedTest]:
        return [LoadedTest(test, lp) for lp in self.packs for test in lp.pack.tests]

    def find(self, key: str) -> LoadedTest | None:
        """By id, then by alias (e.g. an old task name)."""
        tests = self.tests
        for lt in tests:
            if lt.test.id == key:
                return lt
        for lt in tests:
            if key in lt.test.aliases:
                return lt
        return None


def pack_roots(cfg: Config) -> list[tuple[Path, bool]]:
    roots = [(BUILTIN_DIR, True)] if cfg.packs.builtin else []
    roots += [(cfg.resolve_path(d), False) for d in cfg.packs.dirs]
    return roots


def discover(root: Path) -> list[Path]:
    """root/pack.json, or every pack.json below root (hidden folders skipped)."""
    if (root / PACK_FILENAME).is_file():
        return [root / PACK_FILENAME]
    found = []
    for path in sorted(root.rglob(PACK_FILENAME)):
        if not any(part.startswith(".") for part in path.relative_to(root).parts):
            found.append(path)
    return found


def load_pack(path: Path, builtin: bool = False) -> tuple[LoadedPack | None, list[Issue]]:
    where = str(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        return None, [Issue("error", where, f"can't read it: {exc}")]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, [Issue("error", f"{where}:{exc.lineno}:{exc.colno}", f"invalid JSON: {exc.msg}")]
    try:
        pack = Pack.model_validate(data)
    except ValidationError as exc:
        return None, [
            Issue("error", f"{where} {_describe(err['loc'], data)}", clean_message(err["msg"]))
            for err in exc.errors(include_url=False)
        ]
    return LoadedPack(pack, path.resolve(), builtin), []


def _describe(loc: tuple[Any, ...], data: Any) -> str:
    """tests[2].rubric... -> [bash_network_monitor] tests[2].rubric... when the test has an id."""
    text = format_loc(loc) or "(top level)"
    if len(loc) >= 2 and loc[0] == "tests" and isinstance(loc[1], int):
        try:
            test_id = data["tests"][loc[1]].get("id")
        except (KeyError, IndexError, TypeError, AttributeError):
            test_id = None
        if test_id:
            return f"[{test_id}] {text}"
    return text


def load_catalog(cfg: Config) -> Catalog:
    catalog = Catalog()
    for root, builtin in pack_roots(cfg):
        if not root.is_dir():
            if builtin or "dirs" in cfg.packs.model_fields_set:  # the default packs/ folder is optional
                catalog.issues.append(Issue("warning", str(root), "pack folder not found"))
            continue
        paths = discover(root)
        if builtin:  # the original four tests run first, as they always have
            paths.sort(key=lambda p: (p.parent.name != "core", p))
        for path in paths:
            loaded, issues = load_pack(path, builtin)
            catalog.issues.extend(issues)
            if loaded is not None:
                catalog.packs.append(loaded)
    return catalog


def select_tests(
    catalog: Catalog,
    settings: PackSettings,
    *,
    suite: str | None = None,
    tags: Iterable[str] = (),
    ids: Iterable[str] = (),
) -> list[LoadedTest]:
    """Apply suite, tag and id filters (arguments win over [packs] settings)."""
    suite = suite or settings.suite
    include = set(tags) or set(settings.include_tags)
    exclude = set(settings.exclude_tags)
    wanted = set(ids) or set(settings.include_ids)
    selected = []
    for lt in catalog.tests:
        test = lt.test
        if wanted and test.id not in wanted and not wanted & set(test.aliases):
            continue
        if suite and suite not in test.suites:
            continue
        if include and not include & set(test.tags):
            continue
        if exclude & set(test.tags):
            continue
        selected.append(lt)
    return selected
