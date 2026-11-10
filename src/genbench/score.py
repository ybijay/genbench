"""Per-model aggregates and the Overall score, rounded at the same steps as the original script."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from math import fsum
from statistics import mean

from genbench.config import Config
from genbench.results import AttemptRecord, skipped_tests


@dataclass
class TestOutcome:
    test_id: str
    test_name: str
    category: str
    difficulty: str
    accuracy: float
    quality: float
    passed: bool | None
    attempts: int
    errors: int
    category_score: float = 0.0


@dataclass
class ModelSummary:
    model: str
    accuracy: float = 0.0
    categories: dict[str, float] = field(default_factory=dict)
    ttft_s: float = 0.0
    tokens_per_sec: float = 0.0
    prompt_eval_tps: float = 0.0
    avg_time_s: float = 0.0
    total_time_s: float = 0.0
    vram_label: str = "N/A"
    cold_load_s: float = 0.0
    passed: int = 0
    with_mark: int = 0
    errors: int = 0
    tests: list[TestOutcome] = field(default_factory=list)
    terms: dict[str, float] = field(default_factory=dict)
    overall: float = 0.0

    @property
    def pass_rate(self) -> float:
        return round(self.passed / self.with_mark * 100.0, 1) if self.with_mark else 0.0


def _weighted(values: list[tuple[float, float]]) -> float:
    total = fsum(w for _, w in values)
    return fsum(v * w for v, w in values) / total if total else 0.0


def _average(values: list[float]) -> float:
    """fsum, not sum: plain sum gives 22.46 or 22.47 for the same numbers depending on the Python version."""
    return fsum(values) / len(values)


def summarize(records: list[AttemptRecord], cfg: Config) -> list[ModelSummary]:
    """One summary per model with at least one successful attempt, best Overall first.

    A test that any model skipped (its prompt didn't fit the context window) is left out for every model.
    """
    not_counted = skipped_tests(records)
    records = [r for r in records if r.test_id not in not_counted]
    by_model: OrderedDict[str, list[AttemptRecord]] = OrderedDict()
    for record in records:
        by_model.setdefault(record.model, []).append(record)

    weights = cfg.scoring.difficulty_weights
    failed = cfg.run.failed_test_score
    summaries: list[ModelSummary] = []
    for model, rows in by_model.items():
        ok_rows = [r for r in rows if r.ok]
        if not ok_rows:
            continue
        summary = ModelSummary(model=model, errors=len(rows) - len(ok_rows))
        by_test: OrderedDict[str, list[AttemptRecord]] = OrderedDict()
        for row in rows:
            by_test.setdefault(row.test_id, []).append(row)
        for test_id, attempts in by_test.items():
            first = attempts[0]
            acc = mean(a.accuracy if a.ok else failed for a in attempts)
            qual = mean(a.quality if a.ok else failed for a in attempts)
            verdicts = [a.passed for a in attempts if a.ok and a.passed is not None]
            marked = any(a.passed is not None for a in attempts) or any(not a.ok for a in attempts)
            passed = None if not marked else (sum(verdicts) * 2 > len(attempts))
            cat = mean(a.category_points if a.ok else failed for a in attempts)
            summary.tests.append(TestOutcome(test_id, first.test_name, first.category, first.difficulty, acc, qual,
                                             passed, len(attempts), sum(not a.ok for a in attempts), cat))

        summary.accuracy = round(_weighted([(t.accuracy, weights.get(t.difficulty, 1.0)) for t in summary.tests]), 1)
        categories: OrderedDict[str, list[tuple[float, float]]] = OrderedDict()
        for t in summary.tests:
            categories.setdefault(t.category, []).append((t.category_score, weights.get(t.difficulty, 1.0)))
        summary.categories = {c: round(_weighted(v), 1) for c, v in categories.items()}
        summary.passed = sum(1 for t in summary.tests if t.passed)
        summary.with_mark = sum(1 for t in summary.tests if t.passed is not None)

        ttft = [_ttft(r, cfg.scoring.ttft_metric) for r in ok_rows]
        summary.ttft_s = round(_average(ttft), 2)
        summary.tokens_per_sec = round(_average([r.tokens_per_sec or 0.0 for r in ok_rows]), 2)
        summary.prompt_eval_tps = round(_average([r.prompt_eval_tps or 0.0 for r in ok_rows]), 1)
        summary.avg_time_s = round(_average([r.total_s or 0.0 for r in ok_rows]), 2)
        summary.total_time_s = round(fsum(r.total_s or 0.0 for r in ok_rows), 1)
        summary.vram_label = rows[0].vram_label
        summary.cold_load_s = rows[0].cold_load_s or 0.0
        summaries.append(summary)

    _overall(summaries, cfg)
    return sorted(summaries, key=lambda s: s.overall, reverse=True)


def _ttft(record: AttemptRecord, metric: str) -> float:
    if metric == "wall":
        return record.wall_ttft_s or 0.0
    if metric == "first_content":
        return record.first_content_s if record.first_content_s is not None else (record.wall_ttft_s or 0.0)
    return record.warm_ttft_s or 0.0


def unused_categories(summaries: list[ModelSummary], cfg: Config) -> list[str]:
    """Categories that carry weight in Overall but have no test in this run (a run of a few tests, say)."""
    present = {category for s in summaries for category in s.categories}
    weighted = [key.split(".", 1)[1] for key in cfg.scoring.weights if key.startswith("category.")]
    return [category for category in weighted if category not in present]


def _overall(summaries: list[ModelSummary], cfg: Config) -> None:
    if not summaries:
        return
    scoring = cfg.scoring
    max_tps = max(s.tokens_per_sec for s in summaries) or 1.0
    positive = [s.ttft_s for s in summaries if s.ttft_s > 0]
    min_ttft = (min(positive) if positive else 0.0) or 0.05
    # A weighted category with no tests in this run would count as zero for everyone; leave it out instead
    # and scale the other weights back up to 100%.
    unused = {f"category.{category}" for category in unused_categories(summaries, cfg)}
    weights = {key: weight for key, weight in scoring.weights.items() if key not in unused}
    weight_sum = fsum(weights.values())
    for s in summaries:
        if scoring.normalization == "absolute":
            ref_tps = scoring.reference.get("tokens_per_sec", 50.0) or 1.0
            ref_ttft = scoring.reference.get("ttft_s", 0.5) or 0.05
            speed = min(100.0, s.tokens_per_sec / ref_tps * 100.0)
            ttft = min(100.0, (ref_ttft / max(s.ttft_s, 1e-9)) * 100.0)
        else:
            speed = (s.tokens_per_sec / max_tps) * 100.0
            ttft = min(100.0, (min_ttft / max(s.ttft_s, min_ttft)) * 100.0)
        s.terms = {"speed.tokens_per_sec": speed, "speed.ttft": ttft, "accuracy": s.accuracy, "pass_rate": s.pass_rate}
        parts = []
        for key, weight in weights.items():
            if key.startswith("category."):
                value = s.categories.get(key.split(".", 1)[1], 0.0)
            else:
                value = s.terms[key]
            parts.append(weight * value)
        total = fsum(parts)
        if weight_sum > 0 and abs(weight_sum - 1.0) > 1e-9:
            total /= weight_sum
        s.overall = round(total, 1)
