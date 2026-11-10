import json
from pathlib import Path

import pytest
from rich.console import Console

from genbench import cli

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    """Tests must not depend on the machine's settings or on the folder pytest runs from."""
    monkeypatch.chdir(tmp_path)
    for name in ("GENBENCH_CONFIG", "OLLAMA_HOST"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cli, "console", Console(width=200, highlight=False))
    monkeypatch.setattr(cli, "err_console", Console(width=200, highlight=False, stderr=True))


def minimal_test(**overrides):
    test = {
        "id": "says_hi",
        "name": "Says hi",
        "description": "Asks the model to greet the user.",
        "goal": "The answer contains a greeting.",
        "category": "basics",
        "prompt": "Say hi.",
        "rubric": {"accuracy": [{"expect": "Greets the user", "check": "contains", "value": "hi", "points": 100}]},
    }
    test.update(overrides)
    return test


def write_pack(folder: Path, tests, name="mine", **extra) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "pack.json"
    path.write_text(json.dumps({"pack": name, "version": "1.0.0", "tests": tests, **extra}), encoding="utf-8")
    return path
