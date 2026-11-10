"""The optional thinking setting: off, on or a level per model, and one model listed once per setting."""

import json

import pytest
from fake_ollama import FakeOllama
from test_run import ANSWERS
from typer.testing import CliRunner

import genbench.runner as runner_module
from genbench.backends.base import think_value
from genbench.backends.ollama import OllamaBackend
from genbench.cli import app
from genbench.config import Config, ConfigError, load_config

THREE_WAYS = 'models = ["good", { name = "good", thinking = "low" }, { name = "good", thinking = "off" }]\n'


def test_values_and_labels():
    cfg = Config.model_validate({"models": [
        "m", {"name": "m", "thinking": "low"}, {"name": "m", "thinking": "OFF"}, {"name": "m", "thinking": True},
        {"name": "other", "thinking": "default"}, {"name": "third", "thinking": False}]})
    assert [m.thinking for m in cfg.models] == [None, "low", "off", "on", None, "off"]
    assert [m.label for m in cfg.models] == ["m", "m (thinking low)", "m (thinking off)", "m (thinking on)", "other",
                                             "third (thinking off)"]
    assert [think_value(m.thinking) for m in cfg.models] == [None, "low", False, True, None, False]


def test_a_model_may_repeat_only_with_a_different_setting(tmp_path):
    path = tmp_path / "genbench.toml"
    path.write_text('models = ["m", { name = "m", thinking = "off" }, { name = "m", thinking = "off" }]\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="repeated only with a different 'thinking' setting"):
        load_config(path, env={})
    path.write_text('models = [{ name = "m", thinking = "sometimes" }]\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="thinking"):
        load_config(path, env={})


@pytest.fixture
def project(tmp_path, monkeypatch):
    fake = FakeOllama(answers=ANSWERS, models=["good:latest", "lazy:latest"])
    monkeypatch.setattr(runner_module, "make_backend", lambda endpoint: OllamaBackend(endpoint, client=fake.client()))
    (tmp_path / "genbench.toml").write_text(THREE_WAYS + '[run]\nname = "think"\n', encoding="utf-8")
    return tmp_path, fake


def rows(root):
    run_dir = sorted((root / "runs").iterdir())[-1]
    return run_dir, [json.loads(line) for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]


def test_one_model_three_ways_in_one_run(project):
    root, fake = project
    out = CliRunner().invoke(app, ["run", "-t", "bash_network_monitor"])
    assert out.exit_code == 0, out.output
    for header in ("[1/3] good on local", "[2/3] good (thinking low) on local", "[3/3] good (thinking off) on local"):
        assert header in out.output

    chats = [body for path, body in fake.requests if path == "/api/chat" and body.get("stream")]
    assert [body.get("think", "not sent") for body in chats] == ["not sent", "low", False]
    assert {body["model"] for body in chats} == {"good"}  # the server is always asked for the real model name
    assert not [body for path, body in fake.requests if path == "/api/generate"]  # never unloaded between its settings

    run_dir, records = rows(root)
    assert [(r["model"], r["thinking_setting"]) for r in records] == [
        ("good", None), ("good (thinking low)", "low"), ("good (thinking off)", "off")]
    assert [bool(r["thinking"]) for r in records] == [True, True, False]  # off: the fake sends no thinking
    board = (run_dir / "leaderboard.md").read_text(encoding="utf-8")
    assert all(f"**{name}**" in board for name in ("good", "good (thinking low)", "good (thinking off)"))
    saved = {p.name for p in (run_dir / "artifacts").iterdir()}
    assert len(saved) == 3  # each setting keeps its own copy of the generated script
    warmups = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["warmups"]
    assert list(warmups) == ["good", "good (thinking low)", "good (thinking off)"]

    regraded = CliRunner().invoke(app, ["regrade", str(run_dir)])
    assert regraded.exit_code == 0 and "good (thinking off)" in regraded.output


def test_a_setting_the_server_refuses_is_reported_once_and_the_rest_still_run(project):
    root, fake = project
    fake.refuse_think = {"low"}
    out = CliRunner().invoke(app, ["run", "-t", "bash_network_monitor", "-t", "algo_and_tool_dispatch"])
    assert out.exit_code == 0, out.output
    assert out.output.count('The server refused thinking = "low" for this model') == 1
    _, records = rows(root)
    by_model = {}
    for r in records:
        by_model.setdefault(r["model"], []).append(r)
    assert [r["error"] is None for r in by_model["good"]] == [True, True]
    assert all("refused thinking" in r["error"] and "invalid think value" in r["error"] for r in by_model["good (thinking low)"])
    assert [r["passed"] for r in by_model["good (thinking off)"]] == [True, True]
    assert not [b for p, b in fake.requests if p == "/api/chat" and b.get("stream") and b.get("think") == "low"]


def test_picking_by_name_or_by_label(project):
    root, _ = project
    assert "3 model(s) x 1 test(s)" in CliRunner().invoke(app, ["run", "-m", "good", "-t", "bash_network_monitor"]).output
    one = CliRunner().invoke(app, ["run", "-m", "good (thinking off)", "-t", "bash_network_monitor"])
    assert "1 model(s) x 1 test(s)" in one.output and "[1/1] good (thinking off) on local" in one.output


def test_list_models_shows_the_setting(project):
    out = CliRunner().invoke(app, ["list", "models"]).output
    assert "Thinking" in out and "default" in out and "low" in out and "off" in out


def test_a_different_model_in_between_is_still_unloaded(project):
    root, fake = project
    (root / "genbench.toml").write_text('models = ["good", "lazy", { name = "good", thinking = "off" }]\n', encoding="utf-8")
    out = CliRunner().invoke(app, ["run", "-t", "bash_network_monitor"])
    assert out.exit_code == 0, out.output
    assert [body["model"] for path, body in fake.requests if path == "/api/generate"] == ["good", "lazy"]
