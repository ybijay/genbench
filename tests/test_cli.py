import json

from typer.testing import CliRunner

from genbench.cli import app
from genbench.config import load_config

runner = CliRunner()


def invoke(*args, **kwargs):
    return runner.invoke(app, list(args), **kwargs)


def test_validate_with_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = invoke("validate", "--portable")
    assert result.exit_code == 0, result.output
    assert "none found, using defaults" in result.output
    assert "4 pack(s), 14 test(s): 0 error(s)" in result.output
    assert "Validation passed." in result.output


def test_explain_shows_goal_pass_mark_and_expectations():
    result = invoke("explain", "python_streamlit_job_parser")
    assert result.exit_code == 0, result.output
    out = result.output
    assert "Python Coding: Streamlit HTML Job Parser" in out
    assert "Goal:" in out and "BeautifulSoup" in out
    assert "accuracy >= 70 (default from [scoring].pass_mark)" in out
    assert "Code is valid Python" in out and "validator python" in out
    assert "Quality expectations (100 points)" in out
    assert "Write a complete Python script using Streamlit" in out


def test_explain_by_old_name_and_unknown():
    assert invoke("explain", "3. Software Architecture (Local AI Gateway & VRAM Scheduler)").exit_code == 0
    result = invoke("explain", "nope")
    assert result.exit_code == 1 and "No test 'nope'" in result.output


def test_list_commands():
    result = invoke("list", "tests", "--suite", "quick")
    assert result.exit_code == 0 and "4 test(s)" in result.output and "bash_network_monitor" in result.output
    assert "built in" in invoke("list", "packs").output
    assert "localhost:11434" in invoke("list", "endpoints").output


def test_schema_prints_json():
    result = invoke("schema")
    assert result.exit_code == 0
    assert json.loads(result.output)["title"] == "genbench test pack"


def test_bad_override_is_a_config_error():
    result = invoke("--set", "run.repeats=0", "validate")
    assert result.exit_code == 2 and "run.repeats" in result.output


def test_init_then_validate(tmp_path):
    models = tmp_path / "models.txt"
    models.write_text("# mine\nqwen3:8b\ngemma3:4b\n", encoding="utf-8")
    project = tmp_path / "proj"
    result = invoke("init", str(project), "--host", "gpu-box", "--host", "192.168.1.50:11500", "--import-models", str(models), "--no-probe")
    assert result.exit_code == 0, result.output
    cfg = load_config(project / "genbench.toml", env={})
    assert [(e.name, e.url) for e in cfg.endpoints] == [
        ("gpu-box", "http://gpu-box:11434"),
        ("192.168.1.50", "http://192.168.1.50:11500"),
    ]
    assert [(m.name, cfg.endpoint(m.endpoint).name) for m in cfg.models] == [("qwen3:8b", "gpu-box"), ("gemma3:4b", "gpu-box")]
    text = (project / "genbench.toml").read_text(encoding="utf-8")
    assert 'models = ["qwen3:8b", "gemma3:4b"]' in text and "[scoring]" not in text and "[validators]" not in text
    assert (project / "packs").is_dir() and (project / "schemas" / "testpack.schema.json").is_file()
    assert invoke("--config", str(project / "genbench.toml"), "validate").exit_code == 0
    again = invoke("init", str(project), "--no-probe")
    assert again.exit_code == 1 and "already exists" in again.output


def test_new_test_from_flags_then_validate(tmp_path):
    project = tmp_path / "proj"
    assert invoke("init", str(project), "--no-probe").exit_code == 0
    config = str(project / "genbench.toml")
    result = invoke(
        "--config", config, "new-test",
        "--id", "capital_cities", "--name", "Capital cities",
        "--description", "Asks for the capital of Australia.",
        "--goal", "Names Canberra, not Sydney.",
        "--category", "knowledge",
        "--expect", "Names Canberra", "--expect", "Does not say Sydney",
    )
    assert result.exit_code == 0, result.output
    pack = json.loads((project / "packs" / "my-tests" / "pack.json").read_text(encoding="utf-8"))
    assert pack["$schema"] == "../../schemas/testpack.schema.json"
    test = pack["tests"][0]
    assert test["goal"] == "Names Canberra, not Sydney."
    assert [c["expect"] for c in test["rubric"]["accuracy"]] == ["Names Canberra", "Does not say Sydney"]

    checked = invoke("--config", config, "validate")
    assert checked.exit_code == 0 and "placeholder value 'TODO'" in checked.output
    assert invoke("--config", config, "validate", "--strict").exit_code == 1
    assert "Names Canberra" in invoke("--config", config, "explain", "capital_cities").output

    duplicate = invoke("--config", config, "new-test", "--id", "capital_cities", "--name", "x", "--description", "x",
                       "--goal", "x", "--category", "knowledge", "--expect", "x")
    assert duplicate.exit_code == 1 and "already exists" in duplicate.output


def test_new_test_interactive(tmp_path):
    project = tmp_path / "proj"
    assert invoke("init", str(project), "--no-probe").exit_code == 0
    answers = "\n".join([
        "Bad Id", "fizzbuzz_rust", "FizzBuzz in Rust", "Asks for FizzBuzz in Rust.", "Correct FizzBuzz output.",
        "coding", "", "Prints Fizz for multiples of 3", "Prints Buzz for multiples of 5", "",
    ]) + "\n"
    result = invoke("--config", str(project / "genbench.toml"), "new-test", input=answers)
    assert result.exit_code == 0, result.output
    pack = json.loads((project / "packs" / "my-tests" / "pack.json").read_text(encoding="utf-8"))
    assert [c["expect"] for c in pack["tests"][0]["rubric"]["accuracy"]] == [
        "Prints Fizz for multiples of 3",
        "Prints Buzz for multiples of 5",
    ]
