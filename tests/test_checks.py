"""Every check type, the validator policies and pass marks."""

import json
import shutil
import sys

import pytest
from conftest import minimal_test

from genbench.config import Config, PassMark, ValidatorSettings
from genbench.grade.engine import GradeContext, grade_test
from genbench.grade.validators import ValidatorRunner
from genbench.pack.models import BenchTest

DEFAULT_MARK = PassMark(accuracy=70)


def make_test(accuracy, quality="same_as_accuracy", **extra):
    return BenchTest.model_validate(minimal_test(rubric={"accuracy": accuracy, "quality": quality}, **extra))


def grade(test, response, tmp_path, validators=None, thinking="", native=None, force=()):
    settings = validators or Config().validators
    ctx = GradeContext(ValidatorRunner(settings, force), tmp_path / "work", tmp_path, DEFAULT_MARK, "m")
    return grade_test(test, response, ctx, thinking=thinking, native_calls=native)


def one(check, points=100):
    return [{"expect": "it", "points": points, **check}]


def test_text_checks(tmp_path):
    cases = [
        ({"check": "contains", "value": "HELLO", "case_sensitive": False}, "well hello there", True),
        ({"check": "contains", "value": "HELLO"}, "well hello there", False),
        ({"check": "not_contains", "value": "sorry"}, "fine", True),
        ({"check": "regex", "pattern": r"\d+", "min_matches": 3}, "1 2", False),
        ({"check": "regex", "pattern": r"\d+", "min_matches": 3}, "1 2 3", True),
        ({"check": "regex", "pattern": "^b", "flags": "m"}, "a\nb", True),
        ({"check": "not_regex", "pattern": "TODO"}, "done", True),
        ({"check": "detect", "what": "markdown_table"}, "| a | b |\n|---|---|\n| 1 | 2 |", True),
        ({"check": "detect", "what": "markdown_table"}, "a | b", False),
        ({"check": "detect", "what": "shebang"}, "\n#!/bin/sh\necho", True),
        ({"check": "json_valid"}, '{"a": 1}', True),
        ({"check": "json_valid"}, "{a: 1}", False),
        ({"check": "yaml_valid"}, "a: 1\nb: [2, 3]", True),
        ({"check": "yaml_valid"}, "just a sentence", False),
        ({"check": "json_schema", "schema": {"type": "object", "required": ["name"]}}, '{"name": "x"}', True),
        ({"check": "json_schema", "schema": {"type": "object", "required": ["name"]}}, '{"nom": "x"}', False),
    ]
    for check, response, expected in cases:
        result = grade(make_test(one(check)), response, tmp_path)
        assert result.checks[0].met is expected, (check, response)
        assert result.accuracy == (100.0 if expected else 0.0)


def test_composites_and_passed(tmp_path):
    rubric = [
        {"id": "a", "expect": "has a", "check": "contains", "value": "a", "points": 50},
        {"expect": "a and b", "check": "all_of", "points": 25, "of": [
            {"check": "passed", "ref": "a"}, {"check": "contains", "value": "b"}]},
        {"expect": "neither x nor y", "check": "none_of", "points": 25, "of": [
            {"check": "contains", "value": "x"}, {"check": "contains", "value": "y"}]},
    ]
    assert grade(make_test(rubric), "a b", tmp_path).accuracy == 100.0
    assert grade(make_test(rubric), "a y", tmp_path).accuracy == 50.0
    assert grade(make_test(rubric), "c", tmp_path).accuracy == 25.0


def test_graded_checks(tmp_path):
    rubric = [{"expect": "long", "check": "word_count", "tiers": [{"min_words": 10, "points": 40}, {"min_words": 5, "points": 20}]},
              {"expect": "answer", "check": "contains", "value": "yes", "points": 60}]
    assert grade(make_test(rubric), "yes " * 10, tmp_path).accuracy == 100.0
    assert grade(make_test(rubric), "yes " * 6, tmp_path).accuracy == 80.0
    result = grade(make_test(rubric), "no", tmp_path)
    assert result.accuracy == 0.0 and "Missed: long; answer" in result.notes

    quality = [{"expect": "half of accuracy", "check": "scaled_score", "factor": 0.5, "max_points": 50},
               {"expect": "says yes", "check": "contains", "value": "yes", "points": 50}]
    test = make_test(one({"check": "contains", "value": "yes"}), quality)
    assert grade(test, "yes", tmp_path).quality == 100.0
    assert grade(test, "nope", tmp_path).quality == 0.0


def test_scores_scale_to_100_and_round(tmp_path):
    rubric = [{"expect": "a", "check": "contains", "value": "a", "points": 1},
              {"expect": "b", "check": "contains", "value": "b", "points": 2}]
    assert grade(make_test(rubric), "a", tmp_path).accuracy == 33.3


def test_answers(tmp_path):
    numeric = one({"check": "numeric_answer", "expected": 1250, "tolerance": 0.01, "relative": True})
    assert grade(make_test(numeric), "Roughly 1,240 units, so 1,251.", tmp_path).accuracy == 100.0
    assert grade(make_test(numeric), "about 1,100", tmp_path).accuracy == 0.0
    pattern = one({"check": "numeric_answer", "expected": 42, "pattern": r"RESULT=(\d+)"})
    assert grade(make_test(pattern), "RESULT=41 then RESULT=42 and 7", tmp_path).accuracy == 100.0
    choice = one({"check": "choice_answer", "expected": "C"})
    assert grade(make_test(choice), "I think B.\nanswer: c", tmp_path).accuracy == 100.0
    assert "no answer found" in grade(make_test(choice), "C", tmp_path).checks[0].note


def test_constraint(tmp_path):
    rubric = one({"check": "constraint", "bullet_count": 3, "max_words_per_bullet": 4, "banned_words": ["very"],
                  "must_end_with": "?"})
    good = "- one\n- two words\n1. three short words\nReady?"
    assert grade(make_test(rubric), good, tmp_path).accuracy == 100.0
    bad = grade(make_test(rubric), "- this bullet is far too long\n- very\nDone.", tmp_path)
    assert bad.accuracy == 0.0
    note = bad.checks[0].note
    assert "2 bullets" in note and "over 4 words" in note and "'very'" in note and "end with" in note


def test_tool_calls_native_and_text(tmp_path):
    rubric = one({"check": "tool_calls", "source": "native", "max_points": 30, "calls": [
        {"name": "search", "args": {"q": "x"}, "points": 20}, {"name": "open", "points": 10}]}, points=0)
    native = [{"function": {"name": "search", "arguments": {"q": "x", "page": 1}}}, {"function": {"name": "open", "arguments": "{}"}}]
    result = grade(make_test(rubric), "calling tools", tmp_path, native=native)
    assert result.accuracy == 100.0 and "tool calls (2): search, open" in result.checks[0].note
    assert grade(make_test(rubric), "calling tools", tmp_path, native=[]).accuracy == 0.0


def test_check_on_thinking_and_response(tmp_path):
    test = make_test(
        [{"expect": "reasoned", "check": "contains", "value": "because", "on": "thinking", "points": 50},
         {"expect": "code", "check": "contains", "value": "print", "points": 50}],
        extract={"kind": "code_block", "langs": ["python"]},
    )
    result = grade(test, "Sure:\n```python\nprint(1)\n```", tmp_path, thinking="because it is simple")
    assert result.accuracy == 100.0


def test_empty_answer_scores_zero(tmp_path):
    test = make_test(one({"check": "contains", "value": "x"}), extract={"kind": "code_block", "langs": ["python"]})
    result = grade(test, "<think>only thinking</think>", tmp_path)
    assert (result.accuracy, result.quality, result.passed) == (0.0, 0.0, False)
    assert result.notes == "Empty response"
    result = grade(test, "```python\n```", tmp_path)
    assert result.notes == "No code generated"


def test_pass_marks(tmp_path):
    rubric = [{"expect": "a", "check": "contains", "value": "a", "points": 60},
              {"expect": "b", "check": "contains", "value": "b", "points": 40}]
    assert grade(make_test(rubric), "a", tmp_path).passed is False  # 60 < default 70
    assert grade(make_test(rubric), "a b", tmp_path).passed is True
    assert grade(make_test(rubric, **{"pass": {"accuracy": 50}}), "a", tmp_path).passed is True
    measured = grade(make_test(rubric, **{"pass": None}), "a", tmp_path)
    assert measured.passed is None and measured.pass_mark is None


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_bash_validator_really_runs(tmp_path):
    settings = ValidatorSettings.model_validate({})
    runner = ValidatorRunner(settings)
    assert runner.run("bash", "if true; then echo ok; fi", tmp_path).status == "ok"
    broken = runner.run("bash", "if true; then echo ok", tmp_path)
    assert broken.status == "fail" and "FAIL" in broken.detail


def test_python_validator_is_builtin(tmp_path):
    runner = ValidatorRunner(ValidatorSettings.model_validate({}))
    assert runner.run("python", "x = 1", tmp_path).status == "ok"
    assert "SyntaxError" in runner.run("python", "def (", tmp_path).detail


def test_missing_validator_policies(tmp_path):
    rubric = [{"expect": "compiles", "check": "validator", "tool": "zig", "points": 50},
              {"expect": "says main", "check": "contains", "value": "main", "points": 50}]
    for policy, accuracy, skipped in (("skip", 100.0, True), ("fallback", 100.0, True), ("zero", 50.0, False)):
        settings = ValidatorSettings.model_validate({"on_missing": policy, "zig": {"cmd": ["no-such-zig", "{file}"]}})
        result = grade(make_test(rubric), "fn main() {}", tmp_path, validators=settings)
        assert result.accuracy == accuracy, policy
        assert result.checks[0].skipped is skipped
    # A pack-level fallback wins over on_missing.
    with_fallback = [{**rubric[0], "fallback": {"check": "contains", "value": "fn ", "fail_points": 10}}, rubric[1]]
    settings = ValidatorSettings.model_validate({"zig": {"cmd": ["no-such-zig", "{file}"]}})
    assert grade(make_test(with_fallback), "fn main() {}", tmp_path, validators=settings).accuracy == 100.0
    assert grade(make_test(with_fallback), "main()", tmp_path, validators=settings).accuracy == 60.0
    forced = grade(make_test(with_fallback), "main()", tmp_path, force={"python", "zig"}, validators=settings)
    assert "fallback forced" in forced.checks[0].note


def test_validator_findings_and_external(tmp_path):
    lint = tmp_path / "lint.py"
    lint.write_text("import json, sys\nprint(json.dumps([{'line': 1}, {'line': 2}]))\nsys.exit(1)\n", encoding="utf-8")
    settings = ValidatorSettings.model_validate({"lint": {"cmd": [sys.executable, str(lint), "{file}"]}})
    rubric = one({"check": "validator_findings", "tool": "lint", "max_points": 10, "penalty_per_finding": 3}, points=0)
    result = grade(make_test(rubric), "code", tmp_path, validators=settings)
    assert result.checks[0].points == 4.0 and "2 finding(s)" in result.checks[0].note

    grader = tmp_path / "grader.py"
    grader.write_text(
        "import json, sys\nreq = json.load(sys.stdin)\n"
        "print(json.dumps({'score': 0.5 if 'half' in req['response'] else 1.0, 'notes': req['test_id']}))\n",
        encoding="utf-8",
    )
    rubric = one({"check": "external", "cmd": [sys.executable, str(grader)], "max_points": 20}, points=0)
    assert grade(make_test(rubric), "half done", tmp_path).accuracy == 50.0
    full = grade(make_test(rubric), "all done", tmp_path)
    assert full.accuracy == 100.0 and full.checks[0].note == "says_hi"
    broken = one({"check": "external", "cmd": [sys.executable, "-c", "import sys; sys.exit(3)"], "max_points": 20}, points=0)
    assert "grader error: exit 3" in grade(make_test(broken), "x", tmp_path).checks[0].note


def test_artifact_is_saved(tmp_path):
    test = make_test(one({"check": "contains", "value": "print"}),
                     extract={"kind": "code_block", "langs": ["python"], "save_as": "app.py"})
    result = grade(test, "```python\nprint('hi')\n```", tmp_path)
    assert result.artifact_path.read_text(encoding="utf-8") == "print('hi')\n"
    assert json.loads(json.dumps(result.notes))  # notes are plain text


def test_table_checks(tmp_path):
    text = ("Intro\n\n| A | B |\n| :--- | ---: |\n| apple | red |\n| kiwi | green |\n\nMore\n\n"
            "| X | Y |\n|---|---|\n| plum | purple |\n")
    checks = [
        ({"check": "table_row", "match": ["APPLE", "red"]}, True),
        ({"check": "table_row", "match": ["apple", "green"]}, False),  # same row required
        ({"check": "table_row", "match": ["\\| A \\|"]}, False),  # header is not a data row
        ({"check": "no_table_row", "match": ["banana"]}, True),
        ({"check": "no_table_row", "match": ["plum"]}, False),  # rows of every table count
        ({"check": "table_rows", "min_rows": 3, "max_rows": 3}, True),
        ({"check": "table_rows", "min_rows": 1, "all_match": ["red|green"]}, False),  # the plum row has neither
    ]
    for check, expected in checks:
        assert grade(make_test(one(check)), text, tmp_path).checks[0].met is expected, check
    assert grade(make_test(one({"check": "no_table_row", "match": ["x"]})), "no table here", tmp_path).checks[0].note == "no table found"


def test_no_tool_calls(tmp_path):
    banned = one({"check": "no_tool_calls", "calls": [{"name": "rm", "args": {"path": "/"}}, {"name": "reboot"}]})
    allowed = '```json\n[{"name": "rm", "arguments": {"path": "/tmp/x"}}]\n```'
    assert grade(make_test(banned), allowed, tmp_path).accuracy == 100.0
    assert grade(make_test(banned), '[{"name": "reboot", "arguments": {}}]', tmp_path).accuracy == 0.0
    assert grade(make_test(banned), "```json\n[]\n```", tmp_path).accuracy == 100.0
    nothing = grade(make_test(banned), "I would not call anything.", tmp_path)
    assert nothing.accuracy == 0.0 and nothing.checks[0].note == "no tool-call list found"
    native = one({"check": "no_tool_calls", "source": "native", "calls": [{"name": "reboot"}]})
    assert grade(make_test(native), "ok", tmp_path, native=[{"function": {"name": "reboot"}}]).accuracy == 0.0


def test_table_without_separator_row_still_counts(tmp_path):
    text = "| Name | Colour |\n| apple | red |\n| kiwi | green |"
    assert grade(make_test(one({"check": "table_rows", "min_rows": 2, "max_rows": 2})), text, tmp_path).accuracy == 100.0
    assert grade(make_test(one({"check": "table_row", "match": ["kiwi", "green"]})), text, tmp_path).accuracy == 100.0
    assert grade(make_test(one({"check": "detect", "what": "markdown_table"})), text, tmp_path).accuracy == 0.0
