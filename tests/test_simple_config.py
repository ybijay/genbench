"""The short config: a one-line model list, optional sections, and no config at all on a single machine."""

import json

import pytest
from fake_ollama import FakeOllama
from test_run import ANSWERS
from typer.testing import CliRunner

import genbench.backends.probe as probe_module
import genbench.cli as cli_module
import genbench.config as config_module
import genbench.runner as runner_module
from genbench.backends.ollama import OllamaBackend
from genbench.backends.probe import ProbeResult, chat_models, is_embedding_model
from genbench.cli import app
from genbench.config import Config, default_bash_cmd, load_config
from genbench.pack.loader import load_catalog
from genbench.scaffold import init_project


def write(tmp_path, text):
    path = tmp_path / "genbench.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_models_as_one_line(tmp_path):
    cfg = load_config(write(tmp_path, 'models = ["qwen3:8b", "gemma3:12b"]\n'), env={})
    assert [(m.name, m.enabled, m.type) for m in cfg.models] == [("qwen3:8b", True, None), ("gemma3:12b", True, None)]
    assert cfg.endpoints[0].url == "http://localhost:11434" and cfg.endpoint(cfg.models[0].endpoint).name == "local"


def test_names_and_blocks_can_be_mixed(tmp_path):
    text = ('models = ["a:1b", { name = "b:2b", type = "moe", options = { num_ctx = 8192 } }]\n'
            '[[endpoints]]\nname = "box"\nurl = "box"\n')
    cfg = load_config(write(tmp_path, text), env={})
    assert [(m.name, m.type, m.options) for m in cfg.models] == [("a:1b", None, {}), ("b:2b", "moe", {"num_ctx": 8192})]
    blocks = load_config(write(tmp_path, '[[models]]\nname = "a:1b"\n[[models]]\nname = "b:2b"\n'), env={})
    assert [m.name for m in blocks.models] == ["a:1b", "b:2b"]  # the long form still works


def test_the_packs_folder_is_optional_unless_you_name_one(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_catalog(Config()).issues == []  # no packs/ folder here, and no complaint
    named = Config.model_validate({"packs": {"dirs": ["my-packs"]}})
    assert [i.message for i in load_catalog(named).issues] == ["pack folder not found"]


def test_your_own_tests_are_picked_up_without_a_packs_section(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write(tmp_path, 'models = ["m"]\n')
    made = CliRunner().invoke(app, ["new-test", "--id", "mine", "--name", "Mine", "--description", "d", "--goal", "g",
                                    "--category", "coding", "--expect", "says hi", "--prompt", "Say hi"])
    assert made.exit_code == 0, made.output
    listed = CliRunner().invoke(app, ["list", "tests"])
    assert "mine" in listed.output and "my-tests" in listed.output


def test_git_bash_is_found_on_windows(tmp_path, monkeypatch):
    assert default_bash_cmd("linux") == ["bash", "-n", "{file}"]
    monkeypatch.setattr(config_module, "GIT_BASH_PLACES", (str(tmp_path / "nowhere" / "bash.exe"),))
    assert default_bash_cmd("win32") == ["bash", "-n", "{file}"]  # not installed: plain bash, with the usual fallback
    git_bash = tmp_path / "Git" / "bin" / "bash.exe"
    git_bash.parent.mkdir(parents=True)
    git_bash.write_text("", encoding="utf-8")
    monkeypatch.setattr(config_module, "GIT_BASH_PLACES", (str(tmp_path / "nowhere" / "bash.exe"), str(git_bash)))
    assert default_bash_cmd("win32") == [git_bash.as_posix(), "-n", "{file}"]


def test_init_writes_only_what_differs(tmp_path):
    init_project(tmp_path / "local", probe=False)
    text = (tmp_path / "local" / "genbench.toml").read_text(encoding="utf-8")
    assert "[" not in text.replace('# models = ["llama3.2:3b"]', "")  # no sections at all: this machine, default everything
    assert "offers the ones installed on the server" in text
    assert load_config(tmp_path / "local" / "genbench.toml", env={}).models == []

    models = tmp_path / "models.txt"
    models.write_text("a:1b\nb:2b\nc:3b\n", encoding="utf-8")
    init_project(tmp_path / "remote", hosts=["gpu-box"], import_models=models, probe=False)
    text = (tmp_path / "remote" / "genbench.toml").read_text(encoding="utf-8")
    assert 'models = [\n  "a:1b",\n  "b:2b",\n  "c:3b",\n]' in text
    assert '[[endpoints]]\nname = "gpu-box"\nurl = "http://gpu-box:11434"\n' in text
    assert "timeout_s" not in text and "[scoring]" not in text and "[packs]" not in text


def test_init_full_spells_everything_out(tmp_path):
    init_project(tmp_path, hosts=["gpu-box"], probe=False, full=True)
    text = (tmp_path / "genbench.toml").read_text(encoding="utf-8")
    for section in ("[run]", "[[endpoints]]", "[defaults.options]", "[packs]", "[validators]", "[scoring]", "[report]"):
        assert section in text
    load_config(tmp_path / "genbench.toml", env={})


def test_the_long_and_short_forms_mean_the_same(tmp_path):
    long_dir, short_dir = tmp_path / "long", tmp_path / "short"
    models = tmp_path / "models.txt"
    models.write_text("a:1b\nb:2b\n", encoding="utf-8")
    init_project(long_dir, hosts=["gpu-box"], import_models=models, probe=False, full=True)
    init_project(short_dir, hosts=["gpu-box"], import_models=models, probe=False)
    long_cfg = load_config(long_dir / "genbench.toml", env={}).model_dump()
    short_cfg = load_config(short_dir / "genbench.toml", env={}).model_dump()
    long_cfg["run"]["name"] = short_cfg["run"]["name"]  # --full names the run after its folder
    for model in long_cfg["models"]:
        model["endpoint"] = None  # --full names each model's endpoint; the short form means the first one
    assert long_cfg == short_cfg


def test_embedding_models_are_left_out():
    assert is_embedding_model("nomic-embed-text:latest") and is_embedding_model("all-minilm:l6", "bert")
    assert not is_embedding_model("qwen3:8b", "qwen3")
    found = ProbeResult("local", "http://localhost:11434", ["qwen3:8b", "nomic-embed-text:latest", "minilm:l6"],
                        families={"minilm:l6": "bert"})
    assert chat_models(found) == ["qwen3:8b"]


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """One machine, no config file: Ollama on localhost with two chat models and an embedding model."""
    fake = FakeOllama(answers=ANSWERS, models=["good:latest", "lazy:latest"])
    monkeypatch.setattr(runner_module, "make_backend", lambda endpoint: OllamaBackend(endpoint, client=fake.client()))
    state = {"answer": ProbeResult("local", "http://localhost:11434", ["good:latest", "lazy:latest", "nomic-embed-text:latest"])}
    monkeypatch.setattr(probe_module, "probe", lambda endpoint, timeout=3.0: state["answer"])
    return tmp_path, state


def test_no_config_offers_the_installed_models(machine, monkeypatch):
    root, _ = machine
    out = CliRunner().invoke(app, ["run", "--suite", "quick"])
    assert out.exit_code == 1 and "has 2 installed" in out.output and "(1 embedding model left out)" in out.output
    assert "Pass --all-models to run all 2, or --model NAME to pick." in out.output and not (root / "runs").exists()

    monkeypatch.setattr(cli_module, "_interactive", lambda: True)
    declined = CliRunner().invoke(app, ["run", "--suite", "quick"], input="n\n")
    assert declined.exit_code == 0 and "Nothing was run" in declined.output and not (root / "runs").exists()
    accepted = CliRunner().invoke(app, ["run", "--suite", "quick"], input="y\n")
    assert accepted.exit_code == 0, accepted.output
    assert "2 model(s) x 4 test(s)" in accepted.output and "[2/2] lazy:latest on local" in accepted.output


def test_all_models_runs_them_without_asking(machine):
    root, _ = machine
    out = CliRunner().invoke(app, ["run", "--all-models", "-t", "bash_network_monitor"])
    assert out.exit_code == 0, out.output
    run_dir = next((root / "runs").iterdir())
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["model"] for r in rows] == ["good:latest", "lazy:latest"]


def test_no_config_and_no_server_says_what_to_do(machine):
    _, state = machine
    state["answer"] = ProbeResult("local", error="http://localhost:11434: ConnectError")
    out = CliRunner().invoke(app, ["run"])
    assert out.exit_code == 1 and "could not be asked which it has" in out.output and "models = [" in out.output
    state["answer"] = ProbeResult("local", "http://localhost:11434", ["nomic-embed-text:latest"])
    out = CliRunner().invoke(app, ["run"])
    assert out.exit_code == 1 and "has none installed" in out.output and "ollama pull" in out.output


def test_listed_models_are_never_replaced_by_installed_ones(machine):
    root, _ = machine
    write(root, 'models = [{ name = "good", enabled = false }]\n')
    out = CliRunner().invoke(app, ["run", "--all-models"])
    assert out.exit_code == 1 and "enabled = false" in out.output
    write(root, 'models = ["good"]\n')
    out = CliRunner().invoke(app, ["run", "-t", "bash_network_monitor"])
    assert out.exit_code == 0 and "1 model(s) x 1 test(s)" in out.output and "installed" not in out.output
    shown = CliRunner().invoke(app, ["validate"])
    assert "Models:   1 enabled" in shown.output
