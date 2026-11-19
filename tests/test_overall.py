"""The Overall score: right answers first, speed as a small share."""

from genbench.config import Config
from genbench.legacy import ORIGINAL_WEIGHTS, with_original_weights
from genbench.report.markdown import leaderboard_markdown
from genbench.results import AttemptRecord
from genbench.score import summarize


def record(model, test_id, accuracy, passed, tps, ttft, category="coding"):
    return AttemptRecord(run_id="r", model=model, test_id=test_id, test_name=test_id, category=category,
                         difficulty="medium", accuracy=accuracy, quality=100.0, category_score=100.0, passed=passed,
                         total_s=10.0, warm_ttft_s=ttft, tokens_per_sec=tps, pass_mark="accuracy >= 70")


def test_by_default_passing_tests_counts_far_more_than_speed():
    # Like run 20261118-183148: the fast model fails some tests, the slow one passes them all.
    rows = []
    for i in range(4):
        rows.append(record("fast", f"t{i}", 100.0 if i < 2 else 60.0, i < 2, tps=80.0, ttft=8.0))
        rows.append(record("slow", f"t{i}", 100.0, True, tps=20.0, ttft=40.0))
    fast, slow = sorted(summarize(rows, Config()), key=lambda s: s.model)
    assert slow.overall > fast.overall
    assert fast.overall == round(0.45 * 50 + 0.45 * 80 + 0.06 * 100 + 0.04 * 100, 1)
    assert slow.overall == round(0.45 * 100 + 0.45 * 100 + 0.06 * 25 + 0.04 * 20, 1)


def test_speed_separates_models_that_answer_equally_well():
    rows = [record("fast", "t1", 100.0, True, tps=60.0, ttft=10.0), record("slow", "t1", 100.0, True, tps=30.0, ttft=40.0)]
    assert [s.model for s in summarize(rows, Config())] == ["fast", "slow"]


def test_the_leaderboard_says_what_overall_is_made_of():
    rows = [record("m", "t1", 100.0, True, tps=50.0, ttft=1.0)]
    board = leaderboard_markdown(summarize(rows, Config()), rows, Config(), {})
    assert ("Overall weighs tests passed 45%, accuracy 45%, generation speed 6% and time to first token 4%. "
            "Speed is scored against the fastest model in this run.") in board


def test_a_run_with_no_pass_marks_leaves_tests_passed_out():
    rows = [record("m", "t1", 80.0, None, tps=50.0, ttft=1.0)]  # a measured-only test: no PASS or FAIL
    [summary] = summarize(rows, Config())
    assert summary.overall == round((0.45 * 80 + 0.06 * 100 + 0.04 * 100) / 0.55, 1)  # not capped at 55
    board = leaderboard_markdown([summary], rows, Config(), {})
    assert "No test in this run has a pass mark, so Overall leaves tests passed out" in board
    assert "Overall weighs accuracy 81.8%, generation speed 10.9% and time to first token 7.3%." in board


def test_old_script_results_are_scored_with_the_old_scripts_weights():
    assert with_original_weights(Config()).scoring.weights == ORIGINAL_WEIGHTS
    own = Config.model_validate({"scoring": {"weights": {"accuracy": 1.0}}})
    assert with_original_weights(own).scoring.weights == {"accuracy": 1.0}  # a config's own weights win
