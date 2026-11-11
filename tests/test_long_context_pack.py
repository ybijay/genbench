"""The long-context pack: the planted facts give the answers the rubric expects, and answers grade as intended."""

import json
import re
from pathlib import Path

import pytest
from fake_ollama import FakeOllama
from typer.testing import CliRunner

import genbench.runner as runner_module
from genbench.backends.ollama import OllamaBackend
from genbench.cli import app
from genbench.config import Config
from genbench.context import estimate_tokens, fingerprint, generate
from genbench.grade.engine import GradeContext, grade_test
from genbench.grade.validators import ValidatorRunner
from genbench.pack.loader import BUILTIN_DIR, Catalog, LoadedTest, load_pack
from genbench.pack.validate import validate_catalog
from genbench.runner import (
    build_messages,
    context_needed,
    context_shortfall,
    cut_off_warning,
    truncation_warning,
)

PACK = BUILTIN_DIR / "long-context" / "pack.json"
pytestmark = pytest.mark.skipif(not PACK.is_file(), reason="the long-context pack is not present")

ANSWERS = {
    "long_context_16k": {"num_parallel": 3, "changed_by": "ops-bob", "oom_model": "llama3.2-vision:11b",
                         "rate_limited_count": 4},
    "long_context_24k": {"num_parallel": 5, "changed_by": "ops-erin", "oom_model": "mistral-small3.2:24b",
                         "rate_limited_count": 6},
    "long_context_48k": {"num_parallel": 4, "changed_by": "ops-erin", "oom_model": "llama3.2-vision:11b",
                         "rate_limited_count": 9},
}
TALLY = {"bad_gateway_count": 8, "cancelled_input_tokens": 9827, "notebook_embed_count": 3,
         "slowest_request": "r-693757", "worst_health_p95_ms": 2200, "rate_limited_client": None}
# Pinned so a change to the generator (or a Rust port that disagrees) shows up here first.
FINGERPRINTS = {"long_context_16k": ("ce8c9ef26b5afad3", 15764), "long_context_24k": ("c34164013288a33f", 23553),
                "long_context_48k": ("4f9be0281c0f03a9", 47157), "long_context_tally_12k": ("4da084ac576eacac", 11824)}


@pytest.fixture(scope="module")
def pack():
    loaded, issues = load_pack(PACK)
    assert not issues, issues
    return loaded


def find(pack, test_id):
    return next(t for t in pack.pack.tests if t.id == test_id)


def oracle(log: str) -> dict:
    """Answer the four questions by rule, straight from the log."""
    changes = re.findall(r"INFO config num_parallel changed \d+ -> (\d+) by (\S+)", log)
    gave_up = re.findall(r"ERROR loader model=(\S+) load failed: CUDA out of memory.*giving up", log)
    rejected = [line for line in log.splitlines() if "client=ide-hanuman-07 " in line and "status=429" in line]
    return {"num_parallel": int(changes[-1][0]), "changed_by": changes[-1][1], "oom_model": gave_up[-1],
            "rate_limited_count": len(rejected)}


REQUEST = re.compile(r"gateway req=(r-\d+) client=(\S+) model=(\S+) status=(\d+) in=(\d+) out=\d+ ms=(\d+)")


def tally_oracle(log: str) -> dict:
    """Answer the six tally questions by rule, straight from the log."""
    requests = [m.groups() for m in map(REQUEST.search, log.splitlines()) if m]
    rate_limited = sorted({client for _, client, _, status, _, _ in requests if status == "429"})
    return {
        "bad_gateway_count": sum(1 for r in requests if r[3] == "502"),
        "cancelled_input_tokens": sum(int(r[4]) for r in requests if r[3] == "499"),
        "notebook_embed_count": sum(1 for r in requests if r[1] == "notebook-lab-3" and r[2] == "nomic-embed-text"),
        "slowest_request": max(requests, key=lambda r: int(r[5]))[0],
        "worst_health_p95_ms": max(int(v) for v in re.findall(r"health \S+ ok p95_ms=(\d+)", log)),
        "rate_limited_client": rate_limited[0] if rate_limited else None,
    }


def grade(pack, test_id, answer, tmp_path):
    cfg = Config()
    ctx = GradeContext(ValidatorRunner(cfg.validators), tmp_path, pack.dir, cfg.scoring.pass_mark)
    result = grade_test(find(pack, test_id), answer, ctx)
    return result.accuracy, result.quality, result.passed


def fenced(obj) -> str:
    return "```json\n" + json.dumps(obj, indent=2) + "\n```"


def test_pack_is_valid_and_portable(pack):
    cfg = Config.model_validate({"packs": {"builtin": False, "dirs": [str(PACK.parent)]}})
    issues = validate_catalog(Catalog(packs=[pack]), cfg, portable=True)
    assert [i for i in issues if i.level == "error" or "Rust" in i.message] == []


def test_the_tally_test_leaves_a_32k_model_room_to_think(pack):
    # On the 16k version one model re-typed the log while thinking and filled its 32k window before answering.
    test = find(pack, "long_context_tally_12k")
    prompt_tokens = estimate_tokens(build_messages(LoadedTest(test, pack))[0]["content"])
    assert 32768 - prompt_tokens * 1.05 > 19500  # about 20k tokens left, more than the log itself


def test_only_the_48k_test_needs_more_than_a_32k_window(pack):
    needs = {t.id: context_needed(build_messages(LoadedTest(t, pack))) for t in pack.pack.tests}
    assert [test_id for test_id, n in needs.items() if n > 32768] == ["long_context_48k"]
    assert needs["long_context_48k"] < 65536 - 8000  # and fits a 64k window with room to think


@pytest.mark.parametrize("test_id", list(ANSWERS))
def test_logs_are_pinned_and_hold_the_expected_answers(pack, test_id):
    log = generate(find(pack, test_id).context)
    assert (fingerprint(log), estimate_tokens(log)) == FINGERPRINTS[test_id]
    assert oracle(log) == ANSWERS[test_id]


def test_the_tally_log_is_pinned_and_holds_the_expected_answers(pack):
    log = generate(find(pack, "long_context_tally_12k").context)
    assert (fingerprint(log), estimate_tokens(log)) == FINGERPRINTS["long_context_tally_12k"]
    assert tally_oracle(log) == TALLY
    times = sorted((int(m.group(6)) for m in map(REQUEST.search, log.splitlines()) if m), reverse=True)
    assert times[:2] == [9248, 9208]  # the slowest request has a close runner-up, but no tie
    assert sum(1 for line in log.splitlines() if "status=499" in line) == 6 and "status=429" not in log


def test_the_48k_log_has_its_decoys(pack):
    log = generate(find(pack, "long_context_48k").context)
    assert log.count("client=ide-hanuman-70 ") == 3 and log.count("client=ide-hanuman-17 ") == 2  # look-alikes
    assert log.count("client=ide-hanuman-07 ") == 12  # nine rejected, three served
    assert log.count("loaded after retry") == 2 and log.count("giving up after 3 attempts") == 1
    assert log.splitlines().index(next(line for line in log.splitlines() if "keep_alive changed" in line)) > \
        log.splitlines().index(next(line for line in log.splitlines() if "by ops-erin" in line))  # a later, unrelated change


def test_the_24k_log_has_its_decoys(pack):
    log = generate(find(pack, "long_context_24k").context)
    assert log.count("client=ide-hanuman-70 ") == 3  # look-alike client
    assert log.count("client=ide-hanuman-07 ") == 8  # two of its requests succeed
    assert "qwen3.8:27b load failed: CUDA out of memory" in log and "qwen3.8:27b loaded after retry" in log
    assert "num_parallel change 5 -> 8 by ops-frank rejected" in log and "num_ctx changed" in log


def test_facts_are_spread_through_the_log(pack):
    lines = generate(find(pack, "long_context_24k").context).splitlines()
    depths = [i / len(lines) for i, line in enumerate(lines) if "ide-hanuman-07 " in line and "429" in line]
    assert depths[0] < 0.1 and depths[-1] > 0.9


def test_the_prompt_carries_the_log(pack):
    test = find(pack, "long_context_16k")
    content = build_messages(LoadedTest(test, pack))[0]["content"]
    assert "{{context}}" not in content and "<<<LOG\n2026-11-03 " in content
    assert content.count("status=429 in=") == 4 and content.endswith('"rate_limited_count": <integer>}')


GOOD_24K = ANSWERS["long_context_24k"]
CASES_24K = [
    ("perfect", fenced(GOOD_24K), (100.0, 100.0, True)),
    ("after-thinking", "<think>ops-frank's change was rejected</think>\n" + fenced(GOOD_24K), (100.0, 100.0, True)),
    ("numbers-as-strings", fenced({**GOOD_24K, "num_parallel": "5", "rate_limited_count": "6"}), (100.0, 100.0, True)),
    ("model-without-tag", fenced({**GOOD_24K, "oom_model": "mistral-small3.2", "changed_by": "@ops-erin"}),
     (100.0, 100.0, True)),
    ("took-every-decoy", fenced({"num_parallel": 8, "changed_by": "ops-frank", "oom_model": "qwen3.8:27b",
                                 "rate_limited_count": 9}), (0.0, 100.0, False)),
    ("counted-the-look-alike", fenced({**GOOD_24K, "rate_limited_count": 9}), (75.0, 100.0, False)),
    ("last-config-line", fenced({**GOOD_24K, "changed_by": "ops-dave"}), (75.0, 100.0, False)),
    ("prose-around-json", "From the log:\n\n" + fenced(GOOD_24K), (100.0, 50.0, True)),
    ("extra-key", fenced({**GOOD_24K, "notes": "ops-frank was rejected"}), (100.0, 50.0, True)),
    ("no-json", "num_parallel is 5, changed by ops-erin.", (0.0, 0.0, False)),
]


@pytest.mark.parametrize(("answer", "expected"), [c[1:] for c in CASES_24K], ids=[c[0] for c in CASES_24K])
def test_long_context_24k_grading(pack, answer, expected, tmp_path):
    assert grade(pack, "long_context_24k", answer, tmp_path) == expected


def test_long_context_16k_grading(pack, tmp_path):
    good = ANSWERS["long_context_16k"]
    assert grade(pack, "long_context_16k", fenced(good), tmp_path) == (100.0, 100.0, True)
    assert grade(pack, "long_context_16k", fenced({**good, "changed_by": "ops-alice", "num_parallel": 4}), tmp_path)[0] == 50.0


def test_long_context_48k_grading(pack, tmp_path):
    good = ANSWERS["long_context_48k"]
    assert grade(pack, "long_context_48k", fenced(good), tmp_path) == (100.0, 100.0, True)
    decoys = {"num_parallel": 4, "changed_by": "ops-alice", "oom_model": "mistral-small3.2:24b", "rate_limited_count": 14}
    assert grade(pack, "long_context_48k", fenced(decoys), tmp_path) == (25.0, 100.0, False)


TALLY_CASES = [
    ("perfect", fenced(TALLY), (100.0, 100.0, True)),
    ("after-thinking", "<think>counting the 502s again: 8</think>\n" + fenced(TALLY), (100.0, 100.0, True)),
    ("one-count-off-still-passes", fenced({**TALLY, "bad_gateway_count": 7}), (83.0, 100.0, True)),
    ("runner-up-as-slowest", fenced({**TALLY, "slowest_request": "r-973116"}), (83.0, 100.0, True)),
    ("invents-a-rate-limited-client", fenced({**TALLY, "rate_limited_client": "ide-ws-03"}), (84.0, 100.0, True)),
    ("two-wrong-fails", fenced({**TALLY, "bad_gateway_count": 10, "cancelled_input_tokens": 8326}), (66.0, 100.0, False)),
    ("req-id-without-prefix", fenced({**TALLY, "slowest_request": "693757"}), (100.0, 100.0, True)),
    ("req-id-as-number", fenced({**TALLY, "slowest_request": 693757}), (100.0, 70.0, True)),
    ("none-as-text", fenced({**TALLY, "rate_limited_client": "none"}), (100.0, 70.0, True)),
    ("numbers-as-strings", fenced({**TALLY, "bad_gateway_count": "8", "worst_health_p95_ms": "2200"}), (100.0, 70.0, True)),
    ("prose-around-json", "Here are the tallies:\n\n" + fenced(TALLY), (100.0, 60.0, True)),
    ("missing-key", fenced({k: v for k, v in TALLY.items() if k != "rate_limited_client"}), (84.0, 70.0, True)),
    ("all-wrong", fenced({"bad_gateway_count": 0, "cancelled_input_tokens": 0, "notebook_embed_count": 0,
                          "slowest_request": "r-1", "worst_health_p95_ms": 0, "rate_limited_client": "ide-ws-01"}),
     (0.0, 100.0, False)),
]


@pytest.mark.parametrize(("answer", "expected"), [c[1:] for c in TALLY_CASES], ids=[c[0] for c in TALLY_CASES])
def test_long_context_tally_grading(pack, answer, expected, tmp_path):
    assert grade(pack, "long_context_tally_12k", answer, tmp_path) == expected


def test_truncation_warning():
    assert truncation_warning(31500, 32768).startswith("Prompt used 31,500 of 32,768 context tokens")
    assert truncation_warning(24000, 32768) is None and truncation_warning(24000, None) is None


@pytest.fixture
def project(tmp_path, monkeypatch):
    fake = FakeOllama(answers={"gateway log": fenced(ANSWERS["long_context_16k"])}, models=["m1:latest"])
    monkeypatch.setattr(runner_module, "make_backend", lambda endpoint: OllamaBackend(endpoint, client=fake.client()))
    (tmp_path / "genbench.toml").write_text(
        f'[run]\nname = "lc"\n[[endpoints]]\nname = "gpu"\nurl = "gpu-box"\n[[models]]\nname = "m1"\n'
        f'[packs]\nbuiltin = false\ndirs = [{json.dumps(str(PACK.parent.parent))}]\n',
        encoding="utf-8",
    )
    return tmp_path, fake


def results(root: Path) -> list[dict]:
    run_dir = next(p for p in (root / "runs").iterdir())
    return [json.loads(line) for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]


def test_run_sends_the_generated_log(project):
    root, fake = project
    out = CliRunner().invoke(app, ["run", "-t", "long_context_16k"])
    assert out.exit_code == 0, out.output
    sent = next(body for path, body in fake.requests if path == "/api/chat" and body.get("stream"))
    assert sent["messages"][-1]["content"].count("client=ide-hanuman-07 ") == 4
    [record] = results(root)
    assert record["passed"] is True and "context tokens" not in record["notes"]
    assert "Warning: Prompt used" not in out.output


def test_run_warns_when_the_prompt_nearly_fills_the_context(project):
    root, fake = project
    fake.prompt_tokens, fake.context_length = 19500, 20000  # the server counted more tokens than estimated
    out = CliRunner().invoke(app, ["run", "-t", "long_context_16k"])
    assert out.exit_code == 0, out.output
    [record] = results(root)
    assert "Prompt used 19,500 of 20,000 context tokens" in record["notes"]
    assert "Warning: Prompt used 19,500 of 20,000" in out.output

    regraded = CliRunner().invoke(app, ["regrade", str(next((root / "runs").iterdir()))])
    assert regraded.exit_code == 0, regraded.output
    again = [p for p in (root / "runs").iterdir() if "regrade" in p.name][0]
    [record] = [json.loads(line) for line in (again / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert "Prompt used 19,500 of 20,000 context tokens" in record["notes"]  # a regrade keeps the warning


def test_context_needed_and_shortfall(pack):
    messages = build_messages(LoadedTest(find(pack, "long_context_16k"), pack))
    needed = context_needed(messages)
    assert 18000 < needed < 19500  # about 16k of prompt, 5% margin, 2,000 tokens to answer
    assert context_shortfall(messages, 32768) is None and context_shortfall(messages, None) is None
    assert context_shortfall(messages, 16384) == f"Needs a context window of about {needed:,} tokens; this model has 16,384"


def test_a_test_that_does_not_fit_is_skipped_and_not_counted(tmp_path, monkeypatch):
    fake = FakeOllama(answers={"gateway log": fenced(ANSWERS["long_context_16k"])}, models=["big:latest", "small:latest"],
                      context_lengths={"small:latest": 16384})
    monkeypatch.setattr(runner_module, "make_backend", lambda endpoint: OllamaBackend(endpoint, client=fake.client()))
    (tmp_path / "genbench.toml").write_text(
        f'[run]\nname = "skip"\n[[endpoints]]\nname = "gpu"\nurl = "gpu-box"\n'
        f'[[models]]\nname = "big"\n[[models]]\nname = "small"\n'
        f'[packs]\nbuiltin = false\ndirs = [{json.dumps(str(PACK.parent.parent))}]\n',
        encoding="utf-8",
    )
    out = CliRunner().invoke(app, ["run", "-t", "agent_failure_recovery", "-t", "long_context_16k"])
    assert out.exit_code == 0, out.output
    assert "SKIP  Needs a context window of about" in out.output and "this model has 16,384" in out.output
    assert "Not counted (not every model could run them): long_context_16k" in out.output

    run_dir = next((tmp_path / "runs").iterdir())
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    by = {(r["model"], r["test_id"]): r for r in rows}
    assert by[("big", "long_context_16k")]["passed"] is True and by[("big", "long_context_16k")]["skipped"] is None
    skipped = by[("small", "long_context_16k")]
    assert skipped["skipped"].startswith("Needs a context window") and skipped["passed"] is None
    sent = [body["model"] for path, body in fake.requests if path == "/api/chat" and body.get("stream")]
    assert sent.count("small") == 1  # the skipped prompt was never sent

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["not_counted"] == {"long_context_16k": ["small"]}
    assert [(m["model"], m["with_mark"], list(m["categories"])) for m in summary["models"]] == [
        ("big", 1, ["agentic"]), ("small", 1, ["agentic"])]  # both compared on the one test both ran
    board = (run_dir / "leaderboard.md").read_text(encoding="utf-8")
    assert "Not counted in the scores above" in board and "(skipped by small)" in board
    assert "| small | SKIPPED | - | - | Needs a context window" in board and "| big | PASS | 100.0 | 100.0 | none |" in board
    assert ",SKIPPED," in (run_dir / "results.csv").read_text(encoding="utf-8")

    regraded = CliRunner().invoke(app, ["regrade", str(run_dir)])
    assert regraded.exit_code == 0, regraded.output
    again = [p for p in (tmp_path / "runs").iterdir() if "regrade" in p.name][0]
    rows = [json.loads(line) for line in (again / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["skipped"] is not None for r in rows] == [False, False, True, False]  # still skipped, not failed


def test_cut_off_warning():
    full = cut_off_warning(16269, 16499, 32768, "length", answered=False)
    assert full == ("Ran out of context window: 16,269 prompt + 16,499 generated tokens filled the 32,768-token window, "
                    "so it never answered")
    assert cut_off_warning(16269, 16499, 32768, "", answered=True).endswith("so the answer may be cut short")
    assert cut_off_warning(1000, 4096, None, "length", answered=True) == (
        "Stopped at the output limit after 4,096 generated tokens, so the answer may be cut short")
    assert cut_off_warning(16269, 6000, 32768, "stop", answered=True) is None
    slid = cut_off_warning(12470, 40288, 24576, "stop", answered=True)  # nemotron on the tally at a 24k window
    assert slid.startswith("Went past the context window: 12,470 prompt + 40,288 generated tokens is more than the 24,576-token")
    assert slid.endswith("the model lost the start of its prompt")


def test_running_out_of_window_while_thinking_is_reported_as_such(project):
    root, fake = project
    fake.answers = {"gateway log": ""}  # all thinking, no answer
    fake.prompt_tokens, fake.completion_tokens, fake.done_reason = 16269, 16499, "length"
    out = CliRunner().invoke(app, ["run", "-t", "long_context_16k"])
    assert out.exit_code == 0, out.output
    [record] = results(root)
    assert record["passed"] is False and record["done_reason"] == "length" and record["context_tokens"] == 32768
    assert record["notes"].startswith("Empty response | Ran out of context window: 16,269 prompt + 16,499 generated")
    assert "Warning: Ran out of context window" in out.output
    run_dir = next((root / "runs").iterdir())
    assert "| m1 | FAIL | 0.0 | 0.0 | No answer. Ran out of context window" in (run_dir / "leaderboard.md").read_text(encoding="utf-8")

    lines = (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    old = json.loads(lines[0])
    for key in ("warnings", "context_tokens", "done_reason"):  # as saved by an earlier genbench
        old.pop(key)
    old["notes"] = "Empty response"
    (run_dir / "results.jsonl").write_text(json.dumps(old) + "\n", encoding="utf-8")
    regraded = CliRunner().invoke(app, ["regrade", str(run_dir)])
    assert regraded.exit_code == 0, regraded.output
    again = [p for p in (root / "runs").iterdir() if "regrade" in p.name][0]
    [record] = [json.loads(line) for line in (again / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert record["warnings"] and "Ran out of context window" in record["notes"]  # worked out from the saved token counts


def test_a_reply_cut_off_without_its_closing_message_is_reported(project):
    root, fake = project
    fake.answers = {"gateway log": ""}
    fake.send_final = False  # the server dies mid-thought: no closing message, so no token counts
    out = CliRunner().invoke(app, ["run", "-t", "long_context_16k"])
    assert out.exit_code == 0, out.output
    [record] = results(root)
    assert record["passed"] is False and record["prompt_tokens"] == 0 and record["done_reason"] == ""
    assert "stopped mid-reply without its closing message, after about 13 characters of thinking, so it never answered" in record["notes"]
    assert "Warning: The server stopped mid-reply" in out.output


def test_explain_shows_the_context(project):
    out = CliRunner().invoke(app, ["explain", "long_context_24k"])
    assert out.exit_code == 0, out.output
    assert "Context:" in out.output and "19 planted line(s)" in out.output and "{{context}}" in out.output
    assert "Needs:" in out.output and "models with less skip the test" in out.output
    full = CliRunner().invoke(app, ["explain", "long_context_24k", "--full-prompt"])
    assert "ops-frank rejected" in full.output and "{{context}}" not in full.output
