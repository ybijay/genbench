"""genbench clean and genbench list runs: what counts as incomplete, what is safe to delete."""

import json
import os
import stat
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from genbench.cli import app
from genbench.results import remove_run, run_info

NOW = datetime.now()


def make_run(root: Path, name: str, *, status=None, attempts=4, expected=None, finished=True, started=None,
             kind=None, idle=None) -> Path:
    folder = root / "runs" / name
    (folder / "artifacts").mkdir(parents=True)
    meta = {"run_id": name, "started_at": (started or NOW).isoformat(timespec="seconds"),
            "models": [{"name": "m1"}], "tests": {f"t{i}": {} for i in range(4)}, "repeats": 1}
    if status:
        meta["status"] = status
    if expected is not None:
        meta["expected_attempts"] = expected
    if finished:
        meta["finished_at"] = NOW.isoformat(timespec="seconds")
    if kind:
        meta["kind"] = kind
    (folder / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    (folder / "results.jsonl").write_text("{}\n" * attempts, encoding="utf-8")
    (folder / "artifacts" / "app.py").write_text("print(1)\n" * 100, encoding="utf-8")
    if idle:
        stamp = (NOW - idle).timestamp()
        for path in (folder / "run.json", folder / "results.jsonl"):
            os.utime(path, (stamp, stamp))
    return folder


@pytest.fixture
def runs(tmp_path):
    (tmp_path / "genbench.toml").write_text('[run]\noutput_dir = "runs"\n', encoding="utf-8")
    day = timedelta(days=1)
    made = {
        "completed": make_run(tmp_path, "20261101-090000_done", status="completed", started=NOW - 8 * day),
        "legacy_complete": make_run(tmp_path, "20261102-090000_old-done", started=NOW - 6.5 * day),
        "legacy_interrupted": make_run(tmp_path, "20261109-120000_stopped-early", attempts=2, started=NOW - 1 * day),
        "interrupted": make_run(tmp_path, "20261109-130000_ctrl-c", status="interrupted", attempts=1),
        "empty": make_run(tmp_path, "20261109-140000_nothing", attempts=0, finished=False),
        "running": make_run(tmp_path, "20261109-150000_live", status="running", attempts=1, finished=False),
        "stale": make_run(tmp_path, "20261109-100000_killed", status="running", attempts=1, finished=False,
                          idle=timedelta(hours=2)),
        "regrade": make_run(tmp_path, "20261103-090000_regrade-x", kind="regrade of an old CSV", started=NOW - 6 * day),
    }
    not_a_run = tmp_path / "runs" / "notes"
    not_a_run.mkdir()
    (not_a_run / "keep.txt").write_text("mine", encoding="utf-8")
    made["not_a_run"] = not_a_run
    return made


def invoke(*args, **kwargs):
    return CliRunner().invoke(app, list(args), **kwargs)


def remaining(made):
    return {key for key, path in made.items() if path.exists()}


def test_status_of_each_kind_of_run(runs):
    statuses = {key: run_info(path).status for key, path in runs.items() if key != "not_a_run"}
    assert statuses == {
        "completed": "completed", "legacy_complete": "completed", "legacy_interrupted": "interrupted",
        "interrupted": "interrupted", "empty": "empty", "running": "running", "stale": "stopped",
        "regrade": "completed",
    }


def test_list_runs(runs):
    result = invoke("list", "runs")
    assert result.exit_code == 0, result.output
    assert "stopped-early" in result.output and "interrupted" in result.output and "2/4" in result.output
    assert "8 run(s)" in result.output and "notes" not in result.output


def test_default_clean_removes_only_incomplete_runs(runs):
    preview = invoke("clean", "--dry-run")
    assert preview.exit_code == 0, preview.output
    assert "Dry run: would delete 4 run folder(s)" in preview.output
    assert remaining(runs) == set(runs)

    result = invoke("clean", "--yes")
    assert result.exit_code == 0, result.output
    assert "Deleted 4 run folder(s)" in result.output
    assert "Leaving 1 run(s) marked as running" in result.output
    assert remaining(runs) == {"completed", "legacy_complete", "running", "regrade", "not_a_run"}
    assert invoke("clean", "--yes").output.strip().startswith("Leaving 1 run(s)")


def test_clean_asks_first(runs):
    declined = invoke("clean", input="n\n")
    assert declined.exit_code == 0 and "Nothing deleted." in declined.output
    assert remaining(runs) == set(runs)
    accepted = invoke("clean", input="y\n")
    assert accepted.exit_code == 0 and "Deleted 4 run folder(s)" in accepted.output


def test_clean_all_keeps_live_runs_and_other_folders(runs):
    result = invoke("clean", "--all", "--yes")
    assert result.exit_code == 0, result.output
    assert remaining(runs) == {"running", "not_a_run"}


def test_older_than_and_keep(runs):
    assert invoke("clean", "--older-than", "7d", "-y").exit_code == 0
    assert "completed" not in remaining(runs) and "legacy_complete" in remaining(runs)
    assert invoke("clean", "--keep", "1", "-y").exit_code == 0
    left = remaining(runs)
    assert "regrade" in left and "legacy_complete" not in left  # newest completed run kept
    assert "legacy_interrupted" in left  # --keep only touches completed runs
    bad = invoke("clean", "--older-than", "soon")
    assert bad.exit_code == 1 and "can't read the age" in bad.output


def test_named_folders(runs):
    result = invoke("clean", runs["completed"].name, "-y")
    assert result.exit_code == 0, result.output
    assert "completed" not in remaining(runs)
    refused = invoke("clean", str(runs["not_a_run"]), "-y")
    assert refused.exit_code == 1 and "isn't a genbench run folder" in refused.output
    assert runs["not_a_run"].exists()
    with pytest.raises(ValueError):
        remove_run(runs["not_a_run"])


def test_nothing_to_clean(tmp_path):
    (tmp_path / "genbench.toml").write_text('[run]\noutput_dir = "runs"\n', encoding="utf-8")
    result = invoke("clean")
    assert result.exit_code == 0 and "Nothing to clean." in result.output


def test_read_only_files_are_removed(runs):
    target = runs["interrupted"] / "artifacts" / "app.py"
    target.chmod(stat.S_IREAD)
    remove_run(runs["interrupted"])
    assert not runs["interrupted"].exists()
