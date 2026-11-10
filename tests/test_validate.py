"""Pack validation: everything a test author can get wrong is reported with where and why."""

from pathlib import Path

from conftest import minimal_test, write_pack

from genbench.config import Config, PassMark
from genbench.pack.loader import load_catalog
from genbench.pack.validate import validate_catalog


def run(folder: Path, *, portable: bool = False, weights=None):
    data = {"packs": {"builtin": False, "dirs": [str(folder)]}}
    if weights is not None:
        data["scoring"] = {"weights": weights}
    cfg = Config.model_validate(data)
    catalog = load_catalog(cfg)
    return catalog, validate_catalog(catalog, cfg, portable=portable)


def errors(issues):
    return [f"{i.where} {i.message}" for i in issues if i.level == "error"]


def warnings(issues):
    return [f"{i.where} {i.message}" for i in issues if i.level == "warning"]


def test_minimal_test_is_valid(tmp_path):
    write_pack(tmp_path / "p", [minimal_test()])
    catalog, issues = run(tmp_path, weights={"category.basics": 1})
    assert len(catalog.tests) == 1
    assert issues == []


def test_name_description_and_goal_are_required(tmp_path):
    test = minimal_test(description="   ")
    del test["goal"]
    write_pack(tmp_path / "p", [test])
    catalog, issues = run(tmp_path)
    assert catalog.tests == []
    text = " ".join(errors(issues))
    assert "[says_hi] tests[0].goal" in text and "tests[0].description" in text


def test_every_top_level_check_needs_expect(tmp_path):
    test = minimal_test()
    test["rubric"]["accuracy"].append({"check": "contains", "value": "!", "points": 5})
    write_pack(tmp_path / "p", [test])
    _, issues = run(tmp_path)
    assert any("accuracy[1] (contains) needs an 'expect' line" in e for e in errors(issues))


def test_nested_checks_do_not_need_expect(tmp_path):
    test = minimal_test()
    test["rubric"]["accuracy"] = [
        {"expect": "Greets", "check": "any_of", "points": 100, "of": [
            {"check": "contains", "value": "hi"}, {"check": "contains", "value": "hello"}]}
    ]
    write_pack(tmp_path / "p", [test])
    assert errors(run(tmp_path)[1]) == []


def test_unknown_check_type_and_bad_json(tmp_path):
    test = minimal_test()
    test["rubric"]["accuracy"][0]["check"] = "contians"
    write_pack(tmp_path / "a", [test], name="a")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "pack.json").write_text('{"pack": "b",\n  "version": }', encoding="utf-8")
    _, issues = run(tmp_path)
    text = " ".join(errors(issues))
    assert "'contians'" in text
    assert "pack.json:2:" in text and "invalid JSON" in text


def test_regex_refs_and_tools_are_checked(tmp_path):
    test = minimal_test()
    test["rubric"]["accuracy"] = [
        {"expect": "Bad regex", "check": "regex", "pattern": "(unclosed", "points": 10},
        {"expect": "Ref to a later check", "check": "passed", "ref": "later", "points": 10},
        {"id": "later", "expect": "Unknown tool", "check": "validator", "tool": "cobol", "points": 10},
        {"expect": "Choice needs a group", "check": "choice_answer", "expected": "B", "pattern": "ANSWER: [A-D]", "points": 10},
    ]
    test["rubric"]["quality"] = [
        {"expect": "Scaled score in the right place", "check": "scaled_score", "factor": 0.5, "max_points": 50},
        {"expect": "Ref to an earlier check", "check": "passed", "ref": "later", "points": 50},
    ]
    write_pack(tmp_path / "p", [test])
    text = " ".join(errors(run(tmp_path)[1]))
    assert "doesn't compile" in text
    assert "ref 'later' doesn't name an earlier check" in text
    assert "validator 'cobol' isn't defined" in text
    assert "needs a capture group" in text
    assert "rubric.quality[1]" not in text  # the quality ref is fine


def test_scaled_score_belongs_in_quality(tmp_path):
    test = minimal_test()
    test["rubric"]["accuracy"].append(
        {"expect": "Scaled", "check": "scaled_score", "factor": 0.5, "max_points": 10}
    )
    write_pack(tmp_path / "p", [test])
    assert any("belongs in rubric.quality" in e for e in errors(run(tmp_path)[1]))


def test_duplicate_ids_across_packs(tmp_path):
    write_pack(tmp_path / "a", [minimal_test()], name="a")
    write_pack(tmp_path / "b", [minimal_test(aliases=["old name"])], name="b")
    write_pack(tmp_path / "c", [minimal_test(id="other", aliases=["old name"])], name="a")
    text = " ".join(errors(run(tmp_path)[1]))
    assert "'says_hi' is also used by" in text
    assert "'old name' is also used by" in text
    assert "pack name 'a' is also used by" in text


def test_prompt_sources(tmp_path):
    both = minimal_test(id="both", prompt_file="p.md")
    missing = minimal_test(id="missing", prompt=None, prompt_file="nope.md")
    write_pack(tmp_path / "p", [both, missing])
    text = " ".join(errors(run(tmp_path)[1]))
    assert "exactly one of 'prompt', 'prompt_file' or 'messages' (found prompt, prompt_file)" in text

    write_pack(tmp_path / "p", [missing])
    assert any("prompt_file not found: nope.md" in e for e in errors(run(tmp_path)[1]))


def test_prompt_file_crlf_and_bom(tmp_path):
    folder = tmp_path / "p"
    write_pack(folder, [minimal_test(prompt=None, prompt_file="prompt.md")])
    (folder / "prompt.md").write_bytes("﻿Line one\r\nLine two\r\n".encode())
    catalog, issues = run(tmp_path)
    assert errors(issues) == []
    assert catalog.tests[0].prompt_text() == "Line one\nLine two"


def test_warnings_for_placeholders_portability_and_weights(tmp_path):
    test = minimal_test(prompt="TODO: write it, using {{language}}")
    test["rubric"]["accuracy"] = [
        {"expect": "Placeholder", "check": "contains", "value": "TODO", "points": 50},
        {"expect": "Lookahead", "check": "regex", "pattern": "foo(?=bar)", "points": 50},
        {"expect": "Nested points", "check": "all_of", "points": 10, "of": [{"check": "contains", "value": "x", "points": 5}]},
    ]
    write_pack(tmp_path / "p", [test])
    _, plain = run(tmp_path, weights={"category.coding": 1})
    _, portable = run(tmp_path, portable=True)
    text = " ".join(warnings(plain))
    assert errors(plain) == []
    assert "prompt still contains TODO" in text
    assert "{{language}} but 'vars' doesn't define it" in text
    assert "placeholder value 'TODO'" in text
    assert "points on a check inside all_of are ignored" in text
    assert "'category.coding' matches no loaded test" in text
    assert "Rust" not in text
    assert any("Rust" in w for w in warnings(portable))


def test_missing_validator_tool_is_a_warning(tmp_path):
    test = minimal_test()
    test["rubric"]["accuracy"] = [{"expect": "Compiles", "check": "validator", "tool": "zig", "points": 100}]
    write_pack(tmp_path / "p", [test])
    cfg = Config.model_validate({
        "packs": {"builtin": False, "dirs": [str(tmp_path)]},
        "validators": {"zig": {"cmd": ["zig-not-installed", "ast-check", "{file}"]}},
    })
    issues = validate_catalog(load_catalog(cfg), cfg)
    assert errors(issues) == []
    assert any("'zig-not-installed' for validator 'zig' isn't installed" in w for w in warnings(issues))


def test_measured_only_tests_have_no_pass_mark(tmp_path):
    write_pack(tmp_path / "p", [minimal_test(**{"pass": None}), minimal_test(id="strict", **{"pass": {"quality": 90}})])
    catalog, issues = run(tmp_path)
    assert errors(issues) == []
    default = PassMark(accuracy=70)
    measured, strict = (lt.test for lt in catalog.tests)
    assert measured.pass_mark(default) is None
    assert strict.pass_mark(default).describe() == "quality >= 90"


def test_missing_pack_folder_is_a_warning(tmp_path):
    _, issues = run(tmp_path / "nowhere")
    assert warnings(issues) and "pack folder not found" in warnings(issues)[0]
