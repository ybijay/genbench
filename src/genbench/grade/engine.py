"""Grade one response against a test: scores, pass or fail, and a met-or-missed result per expectation.

Rubric scores are the points earned over the points available (skipped checks left out), scaled to
0-100 and rounded to one decimal, which reproduces the original script's arithmetic for the core pack.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from genbench.config import PassMark
from genbench.grade.extract import Artifact, extract
from genbench.grade.toolcalls import calls_from_text, locate_calls, normalize_calls, score_calls
from genbench.grade.validators import ValidatorRunner
from genbench.pack.models import BenchTest

_FLAGS = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL, "x": re.VERBOSE}
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")
BULLET = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+(.*)$")
DETECTORS = {
    "markdown_table": re.compile(r"^[ \t]*\|?[^\n]*\|[^\n]*\n[ \t]*\|?[ \t]*:?-{3,}:?[ \t]*\|", re.MULTILINE),
    "ascii_diagram": re.compile("(\\+---|\\|---|┌|──|-->|==>|\\[.*?\\].*?>)"),
    "shebang": re.compile(r"\A\s*#!/"),
    "fenced_block": re.compile(r"```"),
}
TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
REPORTED_NOTES = ("validator", "validator_findings", "tool_calls", "external", "word_count", "numeric_answer", "choice_answer")


@dataclass
class CheckResult:
    section: str
    expect: str
    check: str
    points: float
    max_points: float
    met: bool
    skipped: bool = False
    note: str = ""


@dataclass
class Grade:
    accuracy: float
    quality: float
    passed: bool | None
    pass_mark: str | None
    checks: list[CheckResult]
    notes: str
    artifact: Artifact
    artifact_path: Path | None = None
    category_score: float = 0.0  # what the test adds to its category column on the leaderboard


def category_score(mode: str, accuracy: float, quality: float) -> float:
    if mode == "quality":
        return quality
    if mode == "accuracy":
        return accuracy
    return round((accuracy + quality) / 2, 1)


@dataclass
class GradeContext:
    validators: ValidatorRunner
    workdir: Path  # this attempt's artifact folder
    pack_dir: Path
    default_pass_mark: PassMark
    model: str = ""


@dataclass
class _Result:
    passed: bool
    points: float
    skipped: bool = False
    note: str = ""


@lru_cache(maxsize=1024)
def compile_pattern(pattern: str, flags: str = "", engine: str = "re") -> Any:
    value = 0
    for flag in flags:
        value |= _FLAGS[flag]
    if engine == "regex":
        import regex  # optional dependency

        return regex.compile(pattern, value)
    return re.compile(pattern, value)


def grade_test(
    test: BenchTest,
    response: str | None,
    ctx: GradeContext,
    *,
    thinking: str = "",
    native_calls: list[Any] | None = None,
) -> Grade:
    artifact = extract(response, test.extract)
    ctx.workdir.mkdir(parents=True, exist_ok=True)
    saved: Path | None = None
    if test.extract.save_as and artifact.text:
        saved = ctx.workdir / test.extract.save_as
        saved.write_text(artifact.text + "\n", encoding="utf-8", newline="\n")
    mark = test.pass_mark(ctx.default_pass_mark)
    mark_text = mark.describe() if mark else None

    if not artifact.text:
        if not artifact.response:
            note = "Empty response"
        else:
            note = "No code generated" if test.extract.kind == "code_block" else "Nothing to grade"
        results = [
            CheckResult(section, check.expect or "", check.check, 0.0, check.max_score(), False, note=note)
            for section, checks in test.rubric.sections()
            for check in checks
        ]
        return Grade(0.0, 0.0, _passes(mark, 0.0, 0.0), mark_text, results, note, artifact, None)

    evaluator = _Evaluator(test, artifact, thinking or "", native_calls or [], ctx)
    results, accuracy = evaluator.section("accuracy", test.rubric.accuracy)
    evaluator.accuracy = accuracy
    if test.rubric.quality == "same_as_accuracy":
        quality = accuracy
    else:
        quality_results, quality = evaluator.section("quality", test.rubric.quality)
        results += quality_results
    accuracy, quality = round(accuracy, 1), round(quality, 1)
    return Grade(accuracy, quality, _passes(mark, accuracy, quality), mark_text, results, _notes(results), artifact, saved,
                 category_score(test.rubric.category_score, accuracy, quality))


def _passes(mark: PassMark | None, accuracy: float, quality: float) -> bool | None:
    if mark is None:
        return None
    ok = True
    if mark.accuracy is not None:
        ok = ok and accuracy >= mark.accuracy
    if mark.quality is not None:
        ok = ok and quality >= mark.quality
    return ok


def _notes(results: list[CheckResult]) -> str:
    missed = [r.expect for r in results if not r.met and not r.skipped]
    skipped = [r.expect for r in results if r.skipped]
    parts = ["Missed: " + "; ".join(missed)] if missed else [f"All {len(results) - len(skipped)} expectations met"]
    if skipped:
        parts.append("Skipped: " + "; ".join(skipped))
    parts += [r.note for r in results if r.check in REPORTED_NOTES and r.note]
    return " | ".join(parts)


class _Evaluator:
    def __init__(self, test: BenchTest, artifact: Artifact, thinking: str, native_calls: list[Any], ctx: GradeContext):
        self.test = test
        self.artifact = artifact
        self.thinking = thinking
        self.native_calls = native_calls
        self.ctx = ctx
        self.accuracy = 0.0
        self.ids: dict[str, bool] = {}

    # ----------------------------------------------------------------- plumbing

    def section(self, name: str, checks: list[Any]) -> tuple[list[CheckResult], float]:
        results: list[CheckResult] = []
        awarded = available = 0.0
        for check in checks:
            outcome = self.evaluate(check)
            maximum = check.max_score()
            results.append(
                CheckResult(name, check.expect or "", check.check, round(outcome.points, 2), maximum, outcome.passed,
                            outcome.skipped, outcome.note)
            )
            if not outcome.skipped:
                awarded += outcome.points
                available += maximum
        if available <= 0:
            return results, 0.0
        score = awarded if abs(available - 100.0) < 1e-9 else awarded * 100.0 / available
        return results, min(100.0, score)

    def evaluate(self, check: Any) -> _Result:
        outcome: _Result = getattr(self, "_" + check.check)(check)
        if check.id:
            self.ids[check.id] = outcome.passed
        return outcome

    def text(self, check: Any) -> str:
        if check.on == "response":
            return self.artifact.response
        if check.on == "thinking":
            return self.thinking
        return self.artifact.text

    @staticmethod
    def _bool(check: Any, ok: bool, note: str = "") -> _Result:
        return _Result(ok, check.points if ok else check.fail_points, False, note)

    # ----------------------------------------------------------------- text checks

    def _contains(self, c: Any) -> _Result:
        text = self.text(c)
        ok = c.value in text if c.case_sensitive else c.value.lower() in text.lower()
        return self._bool(c, ok)

    def _not_contains(self, c: Any) -> _Result:
        text = self.text(c)
        found = c.value in text if c.case_sensitive else c.value.lower() in text.lower()
        return self._bool(c, not found)

    def _regex(self, c: Any) -> _Result:
        rx = compile_pattern(c.pattern, c.flags, c.engine)
        text = self.text(c)
        if c.min_matches == 1:
            return self._bool(c, rx.search(text) is not None)
        count = sum(1 for _ in rx.finditer(text))
        return self._bool(c, count >= c.min_matches, f"{count} match(es)")

    def _not_regex(self, c: Any) -> _Result:
        return self._bool(c, compile_pattern(c.pattern, c.flags, c.engine).search(self.text(c)) is None)

    def _all_of(self, c: Any) -> _Result:
        outcomes = [self.evaluate(child) for child in c.of]
        return self._bool(c, all(o.passed for o in outcomes))

    def _any_of(self, c: Any) -> _Result:
        outcomes = [self.evaluate(child) for child in c.of]
        return self._bool(c, any(o.passed for o in outcomes))

    def _none_of(self, c: Any) -> _Result:
        outcomes = [self.evaluate(child) for child in c.of]
        return self._bool(c, not any(o.passed for o in outcomes))

    def _passed(self, c: Any) -> _Result:
        return self._bool(c, self.ids.get(c.ref, False))

    def _detect(self, c: Any) -> _Result:
        return self._bool(c, DETECTORS[c.what].search(self.text(c)) is not None)

    def _word_count(self, c: Any) -> _Result:
        words = len(self.text(c).split())
        points = max((tier.points for tier in c.tiers if words >= tier.min_words), default=0.0)
        return _Result(points >= c.max_score(), points, False, f"{words} words")

    def _code_ratio(self, c: Any) -> _Result:
        points = round(min(c.max_points, self.artifact.ratio * c.scale), 1)
        return _Result(points >= c.max_points, points, False, f"code is {self.artifact.ratio:.0%} of the answer")

    # ----------------------------------------------------------------- structured output

    def _json_valid(self, c: Any) -> _Result:
        try:
            json.loads(self.text(c))
        except ValueError as exc:
            return self._bool(c, False, f"invalid JSON: {exc}")
        return self._bool(c, True)

    def _json_schema(self, c: Any) -> _Result:
        import jsonschema

        schema = c.schema_
        if schema is None:
            schema = json.loads((self.ctx.pack_dir / c.schema_file).read_text(encoding="utf-8-sig"))
        try:
            data = json.loads(self.text(c))
        except ValueError as exc:
            return self._bool(c, False, f"invalid JSON: {exc}")
        error = jsonschema.exceptions.best_match(jsonschema.Draft202012Validator(schema).iter_errors(data))
        return self._bool(c, error is None, "" if error is None else f"schema: {error.message[:120]}")

    def _yaml_valid(self, c: Any) -> _Result:
        """Valid YAML that is a mapping or a list (plain prose also parses as YAML, so it doesn't count)."""
        try:
            data = yaml.safe_load(self.text(c))
        except yaml.YAMLError as exc:
            return self._bool(c, False, f"invalid YAML: {str(exc).splitlines()[0]}")
        return self._bool(c, isinstance(data, (dict, list)), "" if isinstance(data, (dict, list)) else "not a mapping or list")

    def _tool_calls(self, c: Any) -> _Result:
        raw = calls_from_text(self.text(c)) if c.source == "json_in_text" else self.native_calls
        calls = normalize_calls(raw)
        points = min(c.max_score(), score_calls(calls, c.calls, c.count_duplicates))
        names = ", ".join(name for name, _ in calls) or "none"
        return _Result(points >= c.max_score(), points, False, f"tool calls ({len(calls)}): {names}")

    def _numeric_answer(self, c: Any) -> _Result:
        text = self.text(c)
        if c.pattern:
            matches = list(compile_pattern(c.pattern).finditer(text))
            raw = matches[-1].group(1) if matches else None
        else:
            numbers = NUMBER.findall(text)
            raw = numbers[-1] if numbers else None
        if raw is None:
            return self._bool(c, False, "no number found")
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            return self._bool(c, False, f"not a number: {raw!r}")
        tolerance = c.tolerance * abs(c.expected) if c.relative else c.tolerance
        return self._bool(c, abs(value - c.expected) <= tolerance + 1e-12, f"answered {value:g}, expected {c.expected:g}")

    def _choice_answer(self, c: Any) -> _Result:
        matches = list(compile_pattern(c.pattern, "" if c.case_sensitive else "i").finditer(self.text(c)))
        if not matches:
            return self._bool(c, False, "no answer found")
        got = matches[-1].group(1).strip()
        ok = got == c.expected if c.case_sensitive else got.lower() == c.expected.lower()
        return self._bool(c, ok, f"answered {got}, expected {c.expected}")

    def _constraint(self, c: Any) -> _Result:
        text = self.text(c)
        words = len(text.split())
        bullets = [m.group(1) for m in (BULLET.match(line) for line in text.splitlines()) if m]
        problems = []
        if c.min_words is not None and words < c.min_words:
            problems.append(f"{words} words, needs {c.min_words}+")
        if c.max_words is not None and words > c.max_words:
            problems.append(f"{words} words, max {c.max_words}")
        if c.bullet_count is not None and len(bullets) != c.bullet_count:
            problems.append(f"{len(bullets)} bullets, needs {c.bullet_count}")
        if c.max_words_per_bullet is not None:
            long = [b for b in bullets if len(b.split()) > c.max_words_per_bullet]
            if long:
                problems.append(f"{len(long)} bullet(s) over {c.max_words_per_bullet} words")
        for word in c.banned_words:
            if re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE):
                problems.append(f"uses '{word}'")
        if c.must_end_with is not None and not text.rstrip().endswith(c.must_end_with):
            problems.append(f"doesn't end with '{c.must_end_with}'")
        return self._bool(c, not problems, "; ".join(problems))

    # ----------------------------------------------------------------- tools

    def _validator_file(self, c: Any) -> str | None:
        return self.test.extract.save_as if c.on == "extracted" else None

    def _validator(self, c: Any) -> _Result:
        outcome = self.ctx.validators.run(c.tool, self.text(c), self.ctx.workdir, self._validator_file(c))
        if outcome.status == "ok":
            return _Result(True, c.points, False, outcome.detail)
        if outcome.status == "fail":
            return _Result(False, c.fail_points, False, outcome.detail)
        if c.fallback is not None:
            fallback = self.evaluate(c.fallback)
            verdict = "passed" if fallback.passed else "failed"
            points = c.points if fallback.passed else c.fallback.fail_points
            return _Result(fallback.passed, points, False, f"{outcome.detail}; fallback check {verdict}")
        if self.ctx.validators.on_missing == "zero":
            return _Result(False, 0.0, False, outcome.detail)
        return _Result(False, 0.0, True, f"{outcome.detail}; points skipped")

    def _validator_findings(self, c: Any) -> _Result:
        outcome = self.ctx.validators.run(c.tool, self.text(c), self.ctx.workdir, self._validator_file(c))
        if outcome.status == "missing":
            if self.ctx.validators.on_missing == "zero":
                return _Result(False, 0.0, False, outcome.detail)
            return _Result(False, 0.0, True, f"{outcome.detail}; points skipped")
        points = max(0.0, c.max_points - outcome.findings * c.penalty_per_finding)
        return _Result(outcome.findings == 0, points, False, f"{c.tool}: {outcome.findings} finding(s)")

    def _external(self, c: Any) -> _Result:
        payload = {
            "test_id": self.test.id,
            "model": self.ctx.model,
            "response": self.artifact.response,
            "thinking": self.thinking,
            "extracted": self.artifact.text,
            "workdir": str(self.ctx.workdir),
        }
        try:
            proc = subprocess.run(
                c.cmd, input=json.dumps(payload), capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=c.timeout_s, cwd=self.ctx.pack_dir,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return _Result(False, 0.0, False, f"grader error: {type(exc).__name__}: {exc}")
        if proc.returncode != 0:
            return _Result(False, 0.0, False, f"grader error: exit {proc.returncode}: {proc.stderr.strip()[:120]}")
        try:
            data = json.loads(proc.stdout)
            points = float(data["points"]) if "points" in data else float(data["score"]) * c.max_points
        except (ValueError, KeyError, TypeError) as exc:
            return _Result(False, 0.0, False, f"grader error: bad output ({exc})")
        points = max(0.0, min(c.max_points, points))
        return _Result(points >= c.max_points, points, False, str(data.get("notes", ""))[:200])

    def _scaled_score(self, c: Any) -> _Result:
        points = min(c.max_points, self.accuracy * c.factor)
        return _Result(points >= c.max_points, points, False, "")

    def _no_tool_calls(self, c: Any) -> _Result:
        if c.source == "native":
            raw, found = self.native_calls, True
        else:
            raw, found = locate_calls(self.text(c))
        if not found:
            return self._bool(c, False, "no tool-call list found")
        offending = []
        for name, args in normalize_calls(raw):
            for banned in c.calls:
                if name == banned.name and isinstance(args, dict) and all(args.get(k) == v for k, v in banned.args.items()):
                    offending.append(name)
                    break
        return self._bool(c, not offending, ("called " + ", ".join(offending)) if offending else "")

    def _table_matches(self, c: Any) -> tuple[list[str], bool]:
        rows = table_rows(self.text(c))
        patterns = [compile_pattern(p, "i") for p in c.match]
        return rows, any(all(p.search(row) for p in patterns) for row in rows)

    def _table_row(self, c: Any) -> _Result:
        rows, hit = self._table_matches(c)
        return self._bool(c, hit, "" if rows else "no table found")

    def _no_table_row(self, c: Any) -> _Result:
        rows, hit = self._table_matches(c)
        if not rows:
            return self._bool(c, False, "no table found")
        return self._bool(c, not hit, "a row matches" if hit else "")

    def _table_rows(self, c: Any) -> _Result:
        rows = table_rows(self.text(c))
        patterns = [compile_pattern(p, "i") for p in c.all_match]
        unmatched = [row for row in rows if not all(p.search(row) for p in patterns)]
        in_range = len(rows) >= c.min_rows and (c.max_rows is None or len(rows) <= c.max_rows)
        note = f"{len(rows)} row(s)" + (f", {len(unmatched)} missing required content" if unmatched else "")
        return self._bool(c, in_range and not unmatched, note)


def table_rows(text: str) -> list[str]:
    """Data rows of every pipe table in the text: header and separator lines left out.

    A table without the |---| separator row still counts (its first line is taken as the header), so a
    formatting slip costs a 'detect: markdown_table' check, not every check that reads the rows.
    """
    rows: list[str] = []
    block: list[str] = []
    for line in [*text.splitlines(), ""]:
        if line.lstrip().startswith("|"):
            block.append(line.strip())
            continue
        if len(block) >= 2:
            body = block[2:] if TABLE_SEPARATOR.match(block[1]) else block[1:]
            rows.extend(row for row in body if not TABLE_SEPARATOR.match(row))
        block = []
    return rows
