from pathlib import Path

import pytest

from genbench.config import ConfigError, find_config, load_config, normalize_url, read_models_file

ROOT = Path(__file__).resolve().parents[1]


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "genbench.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_without_a_file():
    cfg = load_config(None, env={})
    assert [(e.name, e.url) for e in cfg.endpoints] == [("local", "http://localhost:11434")]
    assert cfg.scoring.pass_mark.accuracy == 70
    assert {"python", "bash"} <= set(cfg.validators.tools)
    assert cfg.source is None and cfg.base_dir == Path.cwd()


@pytest.mark.parametrize(
    ("host", "url"),
    [("blueboy", "http://blueboy:11434"), ("0.0.0.0", "http://localhost:11434"), ("0.0.0.0:9999", "http://localhost:9999")],
)
def test_ollama_host_sets_the_default_endpoint(host, url):
    cfg = load_config(None, env={"OLLAMA_HOST": host})
    assert cfg.endpoints[0].url == url


@pytest.mark.parametrize(
    ("raw", "port", "url"),
    [
        ("blueboy", 11434, "http://blueboy:11434"),
        ("http://blueboy:11434/", 11434, "http://blueboy:11434"),
        ("192.168.1.50", 11434, "http://192.168.1.50:11434"),
        ("https://gpu.example.com/v1/", None, "https://gpu.example.com/v1"),
        ("localhost:1234/v1", None, "http://localhost:1234/v1"),
    ],
)
def test_normalize_url(raw, port, url):
    assert normalize_url(raw, port) == url


def test_example_config_loads():
    cfg = load_config(ROOT / "genbench.example.toml", env={})
    assert [m.name for m in cfg.models] == ["llama3.2:3b", "qwen3:8b"] and cfg.endpoints[0].name == "local"
    assert cfg.validators.tools["python"].builtin == "ast"
    assert cfg.validators.tools["bash"].cmd[-2:] == ["-n", "{file}"]
    assert abs(sum(cfg.scoring.weights.values()) - 1.0) < 1e-9


@pytest.mark.skipif(not (ROOT / "genbench.toml").is_file(), reason="no genbench.toml of your own here")
def test_your_own_config_loads():
    cfg = load_config(ROOT / "genbench.toml", env={})  # git-ignored: whatever it holds, it must load
    assert cfg.endpoints and abs(sum(cfg.scoring.weights.values()) - 1.0) < 1e-9


def test_env_and_set_overrides(tmp_path):
    path = write(
        tmp_path,
        '[run]\nrepeats = 1\n[[endpoints]]\nname = "a"\nurl = "host-a"\n'
        '[scoring.weights]\naccuracy = 0.5\n"category.coding" = 0.5\n',
    )
    cfg = load_config(
        path,
        ["run.repeats=3", 'scoring.weights."category.coding"=0.25', "endpoints.0.url=host-b", "run.name=nightly"],
        env={"GENBENCH_RUN__SEED": "7", "GENBENCH_RUN__REPEATS": "2", "GENBENCH_UNRELATED": "x"},
    )
    assert cfg.run.repeats == 3  # --set beats the environment
    assert cfg.run.seed == 7
    assert cfg.run.name == "nightly"
    assert cfg.scoring.weights == {"accuracy": 0.5, "category.coding": 0.25}
    assert cfg.endpoints[0].url == "http://host-b:11434"
    assert cfg.base_dir == tmp_path.resolve()


def test_unknown_keys_are_rejected(tmp_path):
    with pytest.raises(ConfigError, match=r"run\.repeat"):
        load_config(write(tmp_path, "[run]\nrepeat = 2\n"), env={})


def test_model_must_use_a_known_endpoint(tmp_path):
    text = '[[endpoints]]\nname = "a"\nurl = "x"\n[[models]]\nname = "m"\nendpoint = "b"\n'
    with pytest.raises(ConfigError, match="uses endpoint 'b'"):
        load_config(write(tmp_path, text), env={})


def test_bad_values_are_explained(tmp_path):
    with pytest.raises(ConfigError, match=r"run\.repeats"):
        load_config(write(tmp_path, ""), ["run.repeats=0"], env={})
    with pytest.raises(ConfigError, match="unknown weight"):
        load_config(write(tmp_path, '[scoring.weights]\n"speed.vibes" = 1\n'), env={})
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(write(tmp_path, "[run\n"), env={})
    with pytest.raises(ConfigError, match="KEY=VALUE"):
        load_config(None, ["run.repeats"], env={})


def test_byte_order_mark_is_fine(tmp_path):
    path = tmp_path / "genbench.toml"
    path.write_bytes("﻿[run]\nname = 'bom'\n".encode())
    assert load_config(path, env={}).run.name == "bom"


def test_validators_keep_defaults_and_accept_flat_tools(tmp_path):
    text = '[validators]\non_missing = "skip"\nshellcheck = { cmd = ["shellcheck", "{file}"] }\n'
    cfg = load_config(write(tmp_path, text), env={})
    assert cfg.validators.on_missing == "skip"
    assert set(cfg.validators.tools) == {"python", "bash", "shellcheck"}
    with pytest.raises(ConfigError, match="exactly one of 'builtin' or 'cmd'"):
        load_config(write(tmp_path, '[validators]\nruff = { timeout_s = 5 }\n'), env={})


def test_read_models_file_accepts_ollama_list_output(tmp_path):
    path = tmp_path / "models.txt"
    path.write_text(
        "# comment\n\nNAME                ID              SIZE      MODIFIED\n"
        "qwen3:8b            500a1f067a9f    5.2 GB    2 days ago\n"
        "gemma3:4b\nqwen3:8b\n",
        encoding="utf-8",
    )
    assert read_models_file(path) == ["qwen3:8b", "gemma3:4b"]


def test_read_models_file_from_the_old_project():
    old = ROOT.parent / "models.txt"
    if not old.is_file():
        pytest.skip("old models.txt not present")
    assert read_models_file(old) == ["qwen3.6:35b-a3b-coding-mtp-q4_K_M", "ornith-1.5:35b"]


def test_find_config_order(tmp_path):
    (tmp_path / "genbench.toml").write_text("", encoding="utf-8")
    other = tmp_path / "other.toml"
    other.write_text("", encoding="utf-8")
    assert find_config(None, env={}, cwd=tmp_path) == (tmp_path / "genbench.toml").resolve()
    assert find_config(None, env={"GENBENCH_CONFIG": str(other)}, cwd=tmp_path) == other.resolve()
    assert find_config(other, env={}, cwd=tmp_path) == other.resolve()
    assert find_config(None, env={}, cwd=tmp_path / "missing") is None
    with pytest.raises(ConfigError):
        find_config(tmp_path / "nope.toml", env={})
