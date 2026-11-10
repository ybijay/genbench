"""Load the original ollama_local_benchmark.py so tests can compare genbench with its own grader functions."""

import csv
import importlib.util
import sys
import types
from functools import lru_cache
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
OLD_SCRIPT = PROJECT / "ollama_local_benchmark.py"
OLD_CSVS = [PROJECT / "ollama_benchmark_results.csv", PROJECT / "bkup" / "ollama_benchmark_results.csv"]
OLD_LEADERBOARDS = [PROJECT / "ollama_benchmark_leaderboard.md", PROJECT / "bkup" / "ollama_benchmark_leaderboard.md"]
TASK_IDS = {
    "python_coding": "python_streamlit_job_parser",
    "bash_coding": "bash_network_monitor",
    "system_architecture": "ai_gateway_architecture",
    "algo_and_agentic": "algo_and_tool_dispatch",
}


@lru_cache(maxsize=1)
def old_module():
    if not OLD_SCRIPT.is_file():
        return None
    try:
        import requests  # noqa: F401  (the old script needs it at import time)
    except ImportError:
        sys.modules.setdefault("requests", types.ModuleType("requests"))
    spec = importlib.util.spec_from_file_location("old_ollama_local_benchmark", OLD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def old_grade(test_id: str, output: str) -> tuple[float, float]:
    old = old_module()
    if test_id == "python_streamlit_job_parser":
        acc, score, _ = old.evaluate_python_task(output, "model")
    elif test_id == "bash_network_monitor":
        acc, score, _ = old.evaluate_bash_task(output, "model")
    elif test_id == "ai_gateway_architecture":
        acc, score, _ = old.evaluate_architecture_task(output)
    else:
        acc, score, _ = old.evaluate_agentic_hybrid_task(output)
    return acc, score


def saved_outputs() -> list[tuple[str, str, str]]:
    """(test_id, label, output) for every saved answer in the old CSVs that exist here."""
    out = []
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    for path in OLD_CSVS:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                task = next((t for t in TASK_IDS if row["Task"].startswith(_number(t))), None)
                if task and row["Response Time (s)"] != "ERROR":
                    out.append((TASK_IDS[task], f"{path.parent.name}/{row['Model']}", row["Output"]))
    return out


def _number(task: str) -> str:
    return {"python_coding": "1.", "bash_coding": "2.", "system_architecture": "3.", "algo_and_agentic": "4."}[task]


def leaderboard_overall(path: Path) -> list[tuple[str, float]]:
    """[(model, Overall)] in rank order from an old leaderboard.md."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) > 3 and cells[0].startswith("#"):
            rows.append((cells[1].strip("*"), float(cells[2].strip("*"))))
    return rows
