"""End to end: genbench run against a fake Ollama, then report and regrade."""

import csv
import json
from pathlib import Path

import pytest
from fake_ollama import FakeOllama
from test_grading_parity import GOOD_AGENT, GOOD_ARCH, GOOD_PY, GOOD_SH
from typer.testing import CliRunner

import genbench.runner as runner_module
from genbench.backends.ollama import OllamaBackend
from genbench.cli import app

ANSWERS = {"Streamlit": GOOD_PY, "Bash script": GOOD_SH, "inference gateway": GOOD_ARCH, "QuickSort": GOOD_AGENT}


@pytest.fixture
def project(tmp_path, monkeypatch):
    fake = FakeOllama(answers=ANSWERS, models=["good:latest", "lazy:latest"])
    monkeypatch.setattr(runner_module, "make_backend", lambda endpoint: OllamaBackend(endpoint, client=fake.client()))
    (tmp_path / "genbench.toml").write_text(
        '[run]\nname = "test"\n[[endpoints]]\nname = "gpu"\nurl = "gpu-box"\n'
        '[[models]]\nname = "good"\n[[models]]\nname = "lazy"\noptions = { num_ctx = 8192 }\n'
        '[packs]\nsuite = "quick"\n',  # the four original tests
        encoding="utf-8",
    )
    return tmp_path, fake


def run_cli(*args):
    return CliRunner().invoke(app, list(args))


def only_run(root: Path, prefix: str = "") -> Path:
    runs = sorted(p for p in (root / "runs").iterdir() if p.name.split("_", 1)[1].startswith(prefix or "test"))
    assert runs, "no run folder"
    return runs[-1]


def test_run_writes_results_and_reports(project):
    root, fake = project
    fake.answers = ANSWERS
    result = run_cli("run")
    assert result.exit_code == 0, result.output
    assert "[1/2] good on gpu" in result.output and "ready (cold load 6.17s" in result.output
    assert "PASS" in result.output and "Leaderboard" in result.output

    run_dir = only_run(root)
    lines = (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 8
    first = json.loads(lines[0])
    assert first["model"] == "good" and first["test_id"] == "python_streamlit_job_parser"
    assert first["accuracy"] == 100.0 and first["passed"] is True
    assert first["thinking"] == "Let me think." and first["artifact_path"].endswith("app.py")
    assert (run_dir / first["artifact_path"]).is_file()

    board = (run_dir / "leaderboard.md").read_text(encoding="utf-8")
    assert "| Rank | Model | Overall | Passed | Accuracy | Coding | Architecture | Agentic |" in board
    assert "## Results by test" in board and "- **Goal:**" in board and "| good | PASS |" in board
    with (run_dir / "results.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["Task"] == "Python Coding: Streamlit HTML Job Parser" and rows[0]["Result"] == "PASS"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert [m["model"] for m in summary["models"]] == ["good", "lazy"]
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["finished_at"] and meta["attempts"] == 8
    assert meta["status"] == "completed" and meta["expected_attempts"] == 8
    assert meta["warmups"]["good"]["gpu_pct"] == 90.0 and meta["warmups"]["good"]["others"] == ["lazy:latest"]
    assert "Warning: only 90% of the model fits in VRAM" in result.output
    assert "other models are loaded on the server too: lazy:latest" in result.output

    # The second model asked for its own context size.
    lazy_chats = [b for path, b in fake.requests if path == "/api/chat" and b["model"] == "lazy" and b.get("stream")]
    assert all(b["options"]["num_ctx"] == 8192 for b in lazy_chats)


def test_filters_repeats_and_ad_hoc_models(project):
    root, fake = project
    result = run_cli("run", "--test", "algo_and_tool_dispatch", "--model", "good", "--repeats", "2")
    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in (only_run(root) / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(r["model"], r["test_id"], r["repeat"]) for r in records] == [
        ("good", "algo_and_tool_dispatch", 1), ("good", "algo_and_tool_dispatch", 2)]


def test_missing_model_is_recorded_as_errors(project):
    root, fake = project
    result = run_cli("run", "--model", "ghost", "--test", "algo_and_tool_dispatch")
    assert result.exit_code == 0, result.output
    assert "ERROR HTTP 404" in result.output
    record = json.loads((only_run(root) / "results.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["error"].startswith("HTTP 404") and record["passed"] is None
    assert "No model completed a test" in result.output or "no leaderboard" in result.output


def test_unreachable_server_skips_its_models(project, monkeypatch):
    root, _ = project
    down = FakeOllama(down_hosts={"gpu-box"})
    monkeypatch.setattr(runner_module, "make_backend", lambda endpoint: OllamaBackend(endpoint, client=down.client()))
    result = run_cli("run", "--test", "algo_and_tool_dispatch")
    assert result.exit_code == 0, result.output
    assert "Skipping good: can't reach endpoint 'gpu'" in result.output


def test_report_and_regrade_a_run(project):
    root, _ = project
    assert run_cli("run", "--test", "bash_network_monitor").exit_code == 0
    run_dir = only_run(root)
    (run_dir / "leaderboard.md").unlink()
    rebuilt = run_cli("report", str(run_dir))
    assert rebuilt.exit_code == 0, rebuilt.output
    assert (run_dir / "leaderboard.md").is_file()

    regraded = run_cli("regrade", str(run_dir))
    assert regraded.exit_code == 0, regraded.output
    new_dir = only_run(root, "regrade-")
    old = [json.loads(x) for x in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    new = [json.loads(x) for x in (new_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(r["accuracy"], r["quality"]) for r in new] == [(r["accuracy"], r["quality"]) for r in old]


def test_run_refuses_invalid_packs(project):
    root, _ = project
    bad = root / "packs" / "broken"
    bad.mkdir(parents=True)
    (bad / "pack.json").write_text('{"pack": "broken", "version": "1.0.0", "tests": []}', encoding="utf-8")
    result = run_cli("run")
    assert result.exit_code == 1 and "Fix the errors above" in result.output


def test_ctrl_c_stops_cleanly_and_is_recorded(project, monkeypatch):
    root, _ = project
    calls = {"n": 0}
    original = OllamaBackend.chat

    def chat_then_interrupt(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return original(self, *args, **kwargs)

    monkeypatch.setattr(OllamaBackend, "chat", chat_then_interrupt)
    result = run_cli("run")
    assert result.exit_code == 130, result.output
    assert "Interrupted; writing reports" in result.output
    run_dir = only_run(root)
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["status"] == "interrupted" and meta["attempts"] == 1
    assert (run_dir / "leaderboard.md").is_file()
    cleaned = run_cli("clean", "--yes")
    assert cleaned.exit_code == 0 and "Deleted 1 run folder(s)" in cleaned.output
    assert not run_dir.exists()
