"""The Phase 1 gate: re-grading the original script's saved results reproduces its scores and leaderboard."""

import csv
import json
import sys

import pytest
from legacy_ref import OLD_CSVS, OLD_LEADERBOARDS, leaderboard_overall
from typer.testing import CliRunner

from genbench.cli import app

PAIRS = [(c, b) for c, b in zip(OLD_CSVS, OLD_LEADERBOARDS, strict=True) if c.is_file() and b.is_file()]


def regrade(csv_path, out_dir, *extra):
    return CliRunner().invoke(app, ["--set", f"run.output_dir={out_dir.as_posix()}", "regrade", "--from-python-csv", str(csv_path), *extra])


@pytest.mark.skipif(not PAIRS, reason="old results not next to genbench/")
@pytest.mark.parametrize(("csv_path", "board"), PAIRS, ids=[c.parent.name for c, _ in PAIRS])
def test_old_scores_and_leaderboard_are_reproduced(csv_path, board, tmp_path):
    result = regrade(csv_path, tmp_path)
    assert result.exit_code == 0, result.output
    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        rows = sum(1 for _ in csv.DictReader(fh))
    assert f"{rows}/{rows} rows within 0.1 of the original scores." in result.output
    run_dir = next(tmp_path.iterdir())
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    ours = [(m["model"], m["overall"]) for m in summary["models"]]
    assert ours == leaderboard_overall(board)


@pytest.mark.skipif(not PAIRS, reason="old results not next to genbench/")
def test_a_changed_score_fails_the_gate(tmp_path):
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    source = PAIRS[0][0]
    with source.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        rows, fields = list(reader), reader.fieldnames
    rows[0]["Accuracy (0-100)"] = "50.0"
    tampered = tmp_path / "tampered.csv"
    with tampered.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    result = regrade(tampered, tmp_path / "runs")
    assert result.exit_code == 1
    assert f"{len(rows) - 1}/{len(rows)} rows within 0.1" in result.output


def test_not_a_results_file(tmp_path):
    other = tmp_path / "other.csv"
    other.write_text("a,b\n1,2\n", encoding="utf-8")
    result = regrade(other, tmp_path / "runs")
    assert result.exit_code == 1 and "doesn't look like" in result.output
