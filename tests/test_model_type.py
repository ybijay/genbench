"""Dense or MoE next to each model, and the per-model totals line."""

import json
import re

import pytest
from fake_ollama import FakeOllama
from test_run import ANSWERS
from typer.testing import CliRunner

import genbench.runner as runner_module
from genbench.backends.base import detect_model_type
from genbench.backends.ollama import OllamaBackend
from genbench.cli import _duration, app
from genbench.config import Config, ConfigError, load_config

# What Blueboy's Ollama reported for the five models on 2026-11-09 (run.json, warmups).
BLUEBOY = {
    "qwen3.6:35b-a3b-coding-mtp-q4_K_M": ("qwen35moe", "moe"),
    "ornith-1.5:35b": ("qwen35moe", "moe"),
    "muse-glimmer:30b": ("muse-glimmer", "dense"),
    "qwen3.8-64k:27b": ("qwen35", "dense"),
    "gemma4:31b-it-q4_K_M": ("gemma4", "dense"),
}


@pytest.mark.parametrize(("name", "family", "expected"), [(n, f, e) for n, (f, e) in BLUEBOY.items()], ids=list(BLUEBOY))
def test_the_five_blueboy_models(name, family, expected):
    shown = {"details": {"family": family, "families": [family], "parameter_size": "30B"}}
    assert detect_model_type(shown, name) == expected


def test_signs_of_a_mixture_of_experts_model():
    assert detect_model_type({"model_info": {"llama.expert_count": 8}, "details": {"family": "llama"}}, "mixtral:8x7b") == "moe"
    assert detect_model_type({"model_info": {"llama.expert_count": 0}, "details": {"family": "llama"}}, "llama3:8b") == "dense"
    assert detect_model_type({"details": {"family": "Qwen3MoE"}}, "some-model") == "moe"
    assert detect_model_type({"details": {"family": "custom"}}, "custom:30b-a3b-q4") == "moe"  # 3B active parameters
    assert detect_model_type({"details": {"family": "gemma"}}, "gemma:a3b-lookalike-not") == "moe"
    assert detect_model_type({"details": {"family": "llama"}}, "llama3.2-vision:11b") == "dense"
    assert detect_model_type({}, "anything:30b") is None  # the server said nothing, so no label


def test_duration():
    assert [_duration(s) for s in (59.4, 404.84, 3725)] == ["0m 59s", "6m 45s", "1h 02m 05s"]


@pytest.fixture
def project(tmp_path, monkeypatch):
    fake = FakeOllama(answers=ANSWERS, models=["fast:latest", "plain:latest", "told:latest"],
                      families={"fast:latest": "qwen35moe", "plain:latest": "qwen35", "told:latest": "qwen35"})
    monkeypatch.setattr(runner_module, "make_backend", lambda endpoint: OllamaBackend(endpoint, client=fake.client()))
    (tmp_path / "genbench.toml").write_text(
        '[run]\nname = "types"\n[[endpoints]]\nname = "gpu"\nurl = "gpu-box"\n'
        '[[models]]\nname = "fast"\n[[models]]\nname = "plain"\n[[models]]\nname = "told"\ntype = "moe"\n',
        encoding="utf-8",
    )
    return tmp_path, fake


def test_run_shows_the_type_and_a_total_for_each_model(project):
    root, fake = project
    out = CliRunner().invoke(app, ["run", "-t", "bash_network_monitor", "-t", "algo_and_tool_dispatch"])
    assert out.exit_code == 0, out.output
    assert "[1/3] fast on gpu  (MoE)" in out.output
    assert "[2/3] plain on gpu  (Dense)" in out.output
    assert "[3/3] told on gpu  (MoE)" in out.output  # genbench.toml wins over what the server reports
    totals = re.findall(r"^  Total time (\d+\.\d\d)s \| average (\d+\.\d\d) tok/s \| 2 tests run$", out.output, re.M)
    assert len(totals) == 3 and totals[0][1] == "50.00"

    lines = out.output.splitlines()
    first_total = next(i for i, line in enumerate(lines) if line.startswith("  Total time"))
    assert "Algorithmic and Agentic Triage" in lines[first_total - 1] and lines[first_total + 1] == ""  # right after its tests
    run_dir = next((root / "runs").iterdir())
    warmups = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["warmups"]
    assert {name: info["model_type"] for name, info in warmups.items()} == {"fast": "moe", "plain": "dense", "told": "moe"}
    assert sum(1 for path, _ in fake.requests if path == "/api/show") == 3  # asked once per model


def test_total_counts_skips_and_is_absent_when_nothing_ran(project):
    root, fake = project
    fake.context_lengths = {"plain:latest": 2048}  # too small for the prompt plus room to answer
    out = CliRunner().invoke(app, ["run", "-t", "bash_network_monitor"])
    assert out.exit_code == 0, out.output
    assert out.output.count("  Total time") == 2  # 'plain' skipped its only test, so it has no total
    assert "SKIP  Needs a context window" in out.output


def test_type_must_be_dense_or_moe(tmp_path):
    assert Config.model_validate({"models": [{"name": "m", "type": "moe"}]}).models[0].type == "moe"
    (tmp_path / "genbench.toml").write_text('[[models]]\nname = "m"\ntype = "sparse"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="type"):
        load_config(tmp_path / "genbench.toml")
