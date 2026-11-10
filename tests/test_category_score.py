"""Category columns: what each test adds, and who wins a category."""

import pytest

from genbench.config import Config
from genbench.grade.engine import category_score
from genbench.pack.loader import BUILTIN_DIR, load_pack
from genbench.pack.models import Rubric
from genbench.report.markdown import leaderboard_markdown
from genbench.results import AttemptRecord
from genbench.score import summarize


def record(model, test_id, category, accuracy, quality, category_score=None, passed=True, tps=10.0, difficulty="medium"):
    return AttemptRecord(run_id="r", model=model, test_id=test_id, test_name=test_id, category=category,
                         difficulty=difficulty, accuracy=accuracy, quality=quality, category_score=category_score,
                         passed=passed, total_s=1.0, warm_ttft_s=1.0, tokens_per_sec=tps, pass_mark="accuracy >= 100")


def test_modes():
    assert category_score("mean", 75.0, 100.0) == 87.5
    assert category_score("quality", 75.0, 100.0) == 100.0
    assert category_score("accuracy", 75.0, 100.0) == 75.0
    assert Rubric.model_validate({"accuracy": [{"expect": "x", "check": "contains", "value": "x", "points": 1}]}).category_score == "mean"
    with pytest.raises(ValueError):
        Rubric.model_validate({"category_score": "best", "accuracy": [{"expect": "x", "check": "contains", "value": "x", "points": 1}]})


def test_the_core_pack_keeps_the_original_scripts_rule():
    loaded, issues = load_pack(BUILTIN_DIR / "core" / "pack.json")
    assert not issues and {t.rubric.category_score for t in loaded.pack.tests} == {"quality"}


def test_a_wrong_answer_lowers_the_category_column():
    rows = [
        record("fast", "lc_a", "long_context", 100.0, 100.0, 100.0, tps=80.0),
        record("fast", "lc_b", "long_context", 75.0, 100.0, 87.5, passed=False, tps=80.0, difficulty="hard"),
        record("slow", "lc_a", "long_context", 100.0, 100.0, 100.0),
        record("slow", "lc_b", "long_context", 100.0, 100.0, 100.0, difficulty="hard"),
    ]
    cfg = Config()
    by_model = {s.model: s for s in summarize(rows, cfg)}
    assert by_model["fast"].categories == {"long_context": 92.5}  # (100 x 1 + 87.5 x 1.5) / 2.5
    assert by_model["slow"].categories == {"long_context": 100.0}
    board = leaderboard_markdown(summarize(rows, cfg), rows, cfg, {})
    assert "- **Best long context:** `slow` (100.0%, 2/2 passed, 10.0 tok/s)" in board
    assert "Category columns count correct answers as well as quality" in board
    assert "medium x1, hard x1.5" in board


def test_equal_scores_go_to_the_model_that_passed_more_tests():
    rows = [
        record("fast", "t1", "agentic", 90.0, 100.0, 95.0, passed=False, tps=80.0),
        record("careful", "t1", "agentic", 100.0, 90.0, 95.0, passed=True),
    ]
    board = leaderboard_markdown(summarize(rows, Config()), rows, Config(), {})
    assert "- **Best agentic:** `careful` (95.0%, 1/1 passed, 10.0 tok/s)" in board


def test_runs_saved_before_the_field_existed_still_use_quality():
    rows = [record("old", "t1", "agentic", 50.0, 80.0, None)]
    assert summarize(rows, Config())[0].categories == {"agentic": 80.0}


def test_averages_do_not_depend_on_the_python_version():
    ttfts = [1.11, 1.0, 1.34, 2.03, 79.12, 125.43, 3.86, 2.84, 4.6, 3.32]  # plain sum() gives 22.47 on Python 3.10
    rows = [record("m", f"t{i}", "coding", 100.0, 100.0, 100.0) for i in range(10)]
    for row, ttft in zip(rows, ttfts, strict=True):
        row.warm_ttft_s = ttft
    assert summarize(rows, Config())[0].ttft_s == 22.46


def test_a_run_without_a_weighted_category_is_not_scored_zero_for_it():
    cfg = Config()  # weights: accuracy .30, coding .25, architecture .20, tokens/sec .15, TTFT .10
    rows = [record("m", "t1", "coding", 100.0, 100.0, 100.0)]
    [summary] = summarize(rows, cfg)
    assert summary.overall == 100.0  # not 80: no architecture test ran, so its 20% is left out
    board = leaderboard_markdown([summary], rows, cfg, {})
    assert "This run has no architecture tests, so Overall leaves that part out" in board

    rows.append(record("m", "t2", "architecture", 50.0, 50.0, 50.0))
    [summary] = summarize(rows, cfg)
    assert summary.overall == round(0.30 * 75 + 0.25 * 100 + 0.20 * 50 + 0.15 * 100 + 0.10 * 100, 1)
    assert "This run has no" not in leaderboard_markdown([summary], rows, cfg, {})
