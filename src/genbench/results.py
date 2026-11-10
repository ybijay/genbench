"""Run folders and results.jsonl: one line per attempt, the source every report is built from."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import stat
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from genbench.grade.engine import Grade
from genbench.pack.loader import LoadedTest

RESULTS_FILE = "results.jsonl"
RUN_FILE = "run.json"


class CheckRecord(BaseModel):
    section: str
    expect: str
    check: str
    points: float
    max_points: float
    met: bool
    skipped: bool = False
    note: str = ""


class AttemptRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    run_id: str
    model: str  # the name shown in results; includes the thinking setting when one was set
    thinking_setting: str | None = None  # off, on or a level, when the run asked for one
    endpoint: str = ""
    test_id: str
    test_name: str
    pack: str = ""
    pack_version: str = ""
    test_hash: str = ""
    category: str
    difficulty: str = "medium"
    repeat: int = 1
    started_at: str = ""
    error: str | None = None
    skipped: str | None = None  # why the test was not run for this model; skipped tests are not scored
    # timing
    total_s: float | None = None
    wall_ttft_s: float | None = None
    warm_ttft_s: float | None = None
    first_content_s: float | None = None
    first_thinking_s: float | None = None
    tokens_per_sec: float | None = None
    prompt_eval_tps: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    done_reason: str = ""
    context_tokens: int | None = None  # the model's context window for this attempt, when known
    warnings: list[str] = Field(default_factory=list)  # e.g. the prompt was cut, or the window ran out
    load_s: float | None = None
    # model on the server
    cold_load_s: float | None = None
    vram_label: str = "N/A"
    # output
    response: str = ""
    thinking: str = ""
    tool_calls: list[Any] = Field(default_factory=list)
    # grade
    accuracy: float = 0.0
    quality: float = 0.0
    category_score: float | None = None  # None in runs saved before this field existed: quality is used
    passed: bool | None = None
    pass_mark: str | None = None
    checks: list[CheckRecord] = Field(default_factory=list)
    notes: str = ""
    artifact_path: str | None = None
    legacy: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.skipped is None

    @property
    def category_points(self) -> float:
        return self.quality if self.category_score is None else self.category_score

    def apply_grade(self, grade: Grade, run_dir: Path | None = None) -> None:
        self.accuracy = grade.accuracy
        self.quality = grade.quality
        self.category_score = grade.category_score
        self.passed = grade.passed
        self.pass_mark = grade.pass_mark
        self.notes = grade.notes
        self.checks = [CheckRecord(**vars(result)) for result in grade.checks]
        path = grade.artifact_path
        if path is not None and run_dir is not None:
            with contextlib.suppress(ValueError):
                path = path.relative_to(run_dir)
        self.artifact_path = path.as_posix() if path is not None else None


def skipped_tests(records: list[AttemptRecord]) -> dict[str, list[str]]:
    """Test id -> the models that skipped it. Such tests are left out of every model's scores, so models are
    compared on the same tests."""
    out: dict[str, list[str]] = {}
    for record in records:
        if record.skipped is not None and record.model not in out.setdefault(record.test_id, []):
            out[record.test_id].append(record.model)
    return out


def test_hash(lt: LoadedTest) -> str:
    """Changes whenever the test's definition or prompt changes, so runs compare like with like."""
    body = json.dumps(lt.test.model_dump(mode="json", by_alias=True), sort_keys=True, ensure_ascii=False)
    prompt = lt.prompt_text() or ""
    return hashlib.sha256((body + "\n" + prompt).encode("utf-8")).hexdigest()[:16]


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_.") or "run"


def new_run_dir(root: Path, name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = root / f"{stamp}_{slug(name)}"
    path, n = base, 2
    while path.exists():
        path, n = Path(f"{base}-{n}"), n + 1
    path.mkdir(parents=True)
    return path


def artifact_dir(run_dir: Path, model: str, test_id: str, repeat: int, repeats: int) -> Path:
    folder = run_dir / "artifacts" / slug(model) / test_id
    return folder / f"repeat{repeat}" if repeats > 1 else folder


class RunStore:
    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir)
        self.results = self.dir / RESULTS_FILE

    def append(self, record: AttemptRecord) -> None:
        with self.results.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(record.model_dump_json() + "\n")

    def write_all(self, records: list[AttemptRecord]) -> None:
        with self.results.open("w", encoding="utf-8", newline="\n") as fh:
            for record in records:
                fh.write(record.model_dump_json() + "\n")

    def load(self) -> list[AttemptRecord]:
        if not self.results.is_file():
            raise FileNotFoundError(f"no {RESULTS_FILE} in {self.dir}")
        lines = self.results.read_text(encoding="utf-8").splitlines()
        return [AttemptRecord.model_validate_json(line) for line in lines if line.strip()]

    def write_meta(self, meta: dict[str, Any]) -> None:
        (self.dir / RUN_FILE).write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

    def read_meta(self) -> dict[str, Any]:
        path = self.dir / RUN_FILE
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


# --------------------------------------------------------------------------- finding and removing runs

STALE_AFTER = timedelta(minutes=30)  # a "running" run untouched this long was killed, not still going
INCOMPLETE = ("interrupted", "failed", "empty", "stopped")


@dataclass
class RunInfo:
    path: Path
    status: str  # completed, interrupted, failed, empty, stopped (killed), running
    kind: str  # run or regrade
    started: datetime
    modified: datetime
    attempts: int
    expected: int | None
    models: int
    size: int


def is_run_dir(path: Path) -> bool:
    """Only folders genbench wrote (they hold run.json or results.jsonl) are ever treated as runs."""
    return path.is_dir() and ((path / RUN_FILE).is_file() or (path / RESULTS_FILE).is_file())


def run_info(path: Path, now: datetime | None = None) -> RunInfo:
    now = now or datetime.now()
    meta: dict[str, Any] = {}
    if (path / RUN_FILE).is_file():
        try:
            meta = json.loads((path / RUN_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
    results = path / RESULTS_FILE
    attempts = 0
    if results.is_file():
        with results.open(encoding="utf-8", errors="replace") as fh:
            attempts = sum(1 for line in fh if line.strip())
    markers = [p for p in (path / RUN_FILE, results) if p.is_file()]
    modified = datetime.fromtimestamp(max(p.stat().st_mtime for p in markers)) if markers else datetime.fromtimestamp(path.stat().st_mtime)
    try:
        started = datetime.fromisoformat(meta["started_at"])
    except (KeyError, TypeError, ValueError):
        started = datetime.fromtimestamp(path.stat().st_mtime)

    expected = meta.get("expected_attempts")
    if expected is None and meta.get("models") and meta.get("tests"):
        expected = len(meta["models"]) * len(meta["tests"]) * int(meta.get("repeats") or 1)
    is_regrade = bool(meta.get("kind") or meta.get("regraded_from")) or "_regrade-" in path.name
    status = meta.get("status")
    if status is None:  # runs written before genbench recorded a status
        if attempts == 0:
            status = "empty"
        elif is_regrade:
            status = "completed"
        elif meta.get("finished_at"):
            status = "completed" if expected is None or attempts >= expected else "interrupted"
        else:
            status = "running"
    if status == "running" and now - modified > STALE_AFTER:
        status = "stopped"
    if status in ("interrupted", "failed") and attempts == 0:
        status = "empty"
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return RunInfo(path, status, "regrade" if is_regrade else "run", started, modified, attempts, expected,
                   len(meta.get("models") or []), size)


def scan_runs(root: Path, now: datetime | None = None) -> list[RunInfo]:
    """Every run folder directly under root, newest first."""
    if not root.is_dir():
        return []
    runs = [run_info(child, now) for child in root.iterdir() if is_run_dir(child)]
    return sorted(runs, key=lambda r: r.started, reverse=True)


def remove_run(path: Path) -> None:
    """Delete a run folder, clearing read-only flags that would stop Windows from deleting files."""

    def retry(func, target, _exc):
        os.chmod(target, stat.S_IWRITE)
        func(target)

    if not is_run_dir(path):
        raise ValueError(f"{path} isn't a genbench run folder")
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=retry)
    else:
        shutil.rmtree(path, onerror=retry)


def parse_age(text: str) -> timedelta:
    """'30d', '12h', '45m' or '2w'."""
    match = re.fullmatch(r"\s*(\d+)\s*([mhdw])\s*", text.lower())
    if not match:
        raise ValueError(f"can't read the age {text!r}; use a number and m, h, d or w, e.g. 30d")
    amount, unit = int(match.group(1)), match.group(2)
    return {"m": timedelta(minutes=amount), "h": timedelta(hours=amount), "d": timedelta(days=amount),
            "w": timedelta(weeks=amount)}[unit]


def human_size(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"
