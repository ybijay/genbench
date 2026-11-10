"""The code-reasoning pack: the expected answers come from really running the code, and answers grade as intended."""

import json
import re

import pytest

from genbench.config import Config
from genbench.grade.engine import GradeContext, grade_test
from genbench.grade.validators import ValidatorRunner
from genbench.pack.loader import BUILTIN_DIR, Catalog, LoadedTest, load_pack
from genbench.pack.validate import validate_catalog

PACK = BUILTIN_DIR / "code-reasoning" / "pack.json"
pytestmark = pytest.mark.skipif(not PACK.is_file(), reason="the code-reasoning pack is not present")


@pytest.fixture(scope="module")
def pack():
    loaded, issues = load_pack(PACK)
    assert not issues, issues
    return loaded


@pytest.fixture(scope="module")
def raw():
    return {t["id"]: t for t in json.loads(PACK.read_text(encoding="utf-8"))["tests"]}


def prompt(pack, test_id) -> str:
    return LoadedTest(next(t for t in pack.pack.tests if t.id == test_id), pack).prompt_text()


def grade(pack, test_id, answer, tmp_path):
    test = next(t for t in pack.pack.tests if t.id == test_id)
    cfg = Config()
    ctx = GradeContext(ValidatorRunner(cfg.validators), tmp_path, pack.dir, cfg.scoring.pass_mark)
    result = grade_test(test, answer, ctx)
    return result.accuracy, result.quality, result.passed


def fenced(obj) -> str:
    return "```json\n" + json.dumps(obj, indent=2) + "\n```"


def test_pack_is_valid_and_portable(pack):
    cfg = Config.model_validate({"packs": {"builtin": False, "dirs": [str(PACK.parent)]}})
    issues = validate_catalog(Catalog(packs=[pack]), cfg, portable=True)
    assert [i for i in issues if i.level == "error" or "Rust" in i.message] == []
    assert {t.category for t in pack.pack.tests} == {"coding"}


# --------------------------------------------------------------------------- code_trace_python

GOOD = {
    "q1": ["a", "c"], "q2": [True, True], "q3": [3, [6], []],
    "q4": [["qwen", "gemma", "mistral"], ["qwen", "gemma", "llama"]],
    "q5": ["gemma", "too-small", "too-small"], "q6": [[16, 4, 1], 10, -11], "q7": [65536, 0.7], "q8": ["ornith", 103],
}
NAIVE = {
    "q1": ["a", "b", "c"], "q2": [False, True], "q3": [3, [6], [[0, 1, 2], [3, 4, 5], [6]]],
    "q4": [["qwen", "gemma", "llama", "mistral"], ["qwen", "gemma", "llama", "mistral"]],
    "q5": ["gemma", "embed", "none"], "q6": [[16, 4, 1], 10, -10], "q7": [32768, 0.0], "q8": ["qwen", 103],
}


def test_the_expected_outputs_are_what_the_script_really_prints(pack, raw):
    code = re.search(r"```python\n(.*?)\n```", prompt(pack, "code_trace_python"), re.S).group(1)
    printed = []
    exec(compile(code, "trace_script", "exec"), {"print": lambda *args: printed.append(args), "__name__": "trace_script"})
    values = [json.loads(json.dumps(args[0] if len(args) == 1 else list(args))) for args in printed]
    assert dict(zip(GOOD, values, strict=True)) == GOOD
    rubric = raw["code_trace_python"]["rubric"]["accuracy"]
    assert [check["schema"]["properties"][key]["enum"][0] for key, check in zip(GOOD, rubric, strict=True)] == values
    assert code.count("# Q") == 8 and sum(check["points"] for check in rubric) == 100


TRACE_CASES = [
    ("perfect", fenced(GOOD), (100.0, 100.0, True)),
    ("after-thinking", "<think>round(102.5) is 102</think>\n" + fenced(GOOD), (100.0, 100.0, True)),
    ("single-values-wrapped", fenced({**GOOD, "q1": [["a", "c"]], "q8": [["ornith", 103]]}), (100.0, 100.0, True)),
    ("one-slip-still-passes", fenced({**GOOD, "q1": NAIVE["q1"]}), (87.5, 100.0, True)),
    ("rounds-half-up", fenced({**GOOD, "q8": NAIVE["q8"]}), (87.5, 100.0, True)),
    ("two-slips-fail", fenced({**GOOD, "q2": NAIVE["q2"], "q6": NAIVE["q6"]}), (75.0, 100.0, False)),
    ("every-naive-answer", fenced(NAIVE), (0.0, 100.0, False)),
    ("prose-around-json", "Tracing each one:\n\n" + fenced(GOOD), (100.0, 50.0, True)),
    ("extra-key", fenced({**GOOD, "notes": "q5: finally wins"}), (100.0, 50.0, True)),
    ("printed-text-instead-of-json", fenced({**GOOD, "q1": "['a', 'c']", "q5": "gemma too-small too-small"}),
     (75.0, 100.0, False)),
]


@pytest.mark.parametrize(("answer", "expected"), [c[1:] for c in TRACE_CASES], ids=[c[0] for c in TRACE_CASES])
def test_code_trace_python(pack, answer, expected, tmp_path):
    assert grade(pack, "code_trace_python", answer, tmp_path) == expected


# --------------------------------------------------------------------------- code_review_planted_bugs

BUGS = {11: "return used_gb / total_gb", 17: 'return sorted(idle, key=lambda m: m["size_gb"])',
        20: "def plan_unloads(models, used_gb, total_gb, now, plan=[]):", 49: "if len(recent) > self.limit:",
        58: "for n in range(1, attempts)]"}
TRAPS = {16: '> IDLE_AFTER_S]', 35: "return -(-len(prompts) // batch_size)", 47: "if now is None else now"}


def module_lines(pack) -> list[str]:
    listing = re.search(r"```text\n(.*?)\n```", prompt(pack, "code_review_planted_bugs"), re.S).group(1).split("\n")
    assert [int(line[:2]) for line in listing] == list(range(1, len(listing) + 1))
    return [line[5:] for line in listing]


def test_the_bugs_and_look_alikes_sit_on_the_lines_the_rubric_names(pack, raw):
    lines = module_lines(pack)
    assert len(lines) == 58
    for number, text in {**BUGS, **TRAPS}.items():
        assert text in lines[number - 1], number
    rubric = raw["code_review_planted_bugs"]["rubric"]["accuracy"]
    accepted = [check["schema"]["contains"]["properties"]["line"]["enum"] for check in rubric[:5]]
    assert [number in lines_ for number, lines_ in zip(BUGS, accepted, strict=True)] == [True] * 5
    assert rubric[5]["schema"]["not"]["contains"]["properties"]["line"]["enum"][:3] == list(TRAPS)
    assert not {n for lines_ in accepted for n in lines_} & set(TRAPS)


def test_the_planted_bugs_are_real_and_the_look_alikes_are_correct(pack):
    ns: dict = {"__name__": "review_module"}
    exec(compile("\n".join(module_lines(pack)), "review_module", "exec"), ns)
    models = [{"name": "big", "size_gb": 8, "last_used": 0}, {"name": "small", "size_gb": 1, "last_used": 0},
              {"name": "busy", "size_gb": 4, "last_used": 995}]

    assert ns["vram_pct"](12, 24) == 0.5  # line 11: a fraction, not 50
    assert [m["name"] for m in ns["idle_models"](models, 1000)] == ["small", "big"]  # line 17: smallest first
    assert ns["plan_unloads"](models, 23.6, 24, 1000) == []  # and line 11 means nothing is ever unloaded
    ns["vram_pct"] = lambda used, total: used / total * 100  # with line 11 fixed, line 20 shows
    assert ns["plan_unloads"](models, 23.6, 24, 1000) == ["small", "big"]
    assert ns["plan_unloads"](models, 23.6, 24, 1000) == ["small", "big", "small", "big"]  # line 20: the plan leaks
    limiter = ns["RateLimiter"](2)
    assert [limiter.allow("c", now=t) for t in (0, 1, 2, 3)] == [True, True, True, False]  # line 49: 3 allowed, limit 2
    assert ns["retry_delays"](1, 3) == [2, 4]  # line 58: no first delay, and one too few

    assert ns["idle_models"]([{"name": "edge", "size_gb": 1, "last_used": 400}], 1000) == []  # line 16: 600 s isn't "more than"
    assert [ns["batches_needed"]([0] * n, 3) for n in (0, 6, 7)] == [0, 2, 3]  # line 35: ceiling division
    assert limiter.allow("d", now=0) and limiter.seen["d"] == [0]  # line 47: now=0 is respected


def findings(*lines):
    return [{"line": n, "problem": "does not match its docstring", "fix": "corrected line"} for n in lines]


REVIEW_CASES = [
    ("perfect", fenced(findings(11, 17, 20, 49, 58)), (100.0, 100.0, True)),
    ("cites-the-function-instead", fenced(findings(9, 14, 26, 49, 56)), (100.0, 100.0, True)),
    ("one-harmless-extra", fenced(findings(11, 17, 20, 48, 49, 58)), (100.0, 100.0, True)),
    ("misses-the-rate-limit", fenced(findings(11, 17, 20, 58)), (85.0, 100.0, True)),
    ("flags-a-look-alike", fenced(findings(11, 16, 17, 20, 49, 58)), (85.0, 100.0, True)),
    ("misses-one-and-flags-a-look-alike", fenced(findings(11, 17, 20, 35, 58)), (70.0, 100.0, False)),
    ("flags-everything", fenced(findings(11, 16, 17, 20, 35, 47, 49, 58)), (75.0, 100.0, False)),
    ("only-the-look-alikes", fenced(findings(16, 35, 47)), (10.0, 100.0, False)),
    ("nothing-found", fenced([]), (0.0, 50.0, False)),
    ("lines-as-text", fenced([{**f, "line": str(f["line"])} for f in findings(11, 17, 20, 49, 58)]), (100.0, 50.0, True)),
    ("wrapped-in-object", fenced({"defects": findings(11, 17, 20, 49, 58)}), (0.0, 50.0, False)),
    ("prose-around-json", "Five defects:\n\n" + fenced(findings(11, 17, 20, 49, 58)), (100.0, 50.0, True)),
]


@pytest.mark.parametrize(("answer", "expected"), [c[1:] for c in REVIEW_CASES], ids=[c[0] for c in REVIEW_CASES])
def test_code_review_planted_bugs(pack, answer, expected, tmp_path):
    assert grade(pack, "code_review_planted_bugs", answer, tmp_path) == expected
