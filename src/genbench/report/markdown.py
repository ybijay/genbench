"""leaderboard.md: today's ranked table and category winners, plus what each test asked and how each model did."""

from __future__ import annotations

from collections import OrderedDict

from genbench.config import Config
from genbench.results import AttemptRecord, skipped_tests
from genbench.score import ModelSummary, unused_categories


def _cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _title(category: str) -> str:
    return category.replace("_", " ").replace("-", " ").title()


CATEGORY_NOTE = (
    "Category columns count correct answers as well as quality: each test adds the average of its accuracy and "
    "quality scores, unless the test picks one of them (the core pack uses quality alone, as the original script did)."
)


def _passed_in(summary: ModelSummary, category: str) -> tuple[int, int]:
    """(tests passed, tests with a pass mark) for one category."""
    tests = [t for t in summary.tests if t.category == category and t.passed is not None]
    return sum(1 for t in tests if t.passed), len(tests)


def ordered_categories(summaries: list[ModelSummary], cfg: Config) -> list[str]:
    seen: list[str] = []
    for key in cfg.scoring.weights:
        if key.startswith("category."):
            seen.append(key.split(".", 1)[1])
    for s in summaries:
        for category in s.categories:
            if category not in seen:
                seen.append(category)
    present = {c for s in summaries for c in s.categories}
    return [c for c in seen if c in present]


def leaderboard_markdown(summaries: list[ModelSummary], records: list[AttemptRecord], cfg: Config, meta: dict) -> str:
    categories = ordered_categories(summaries, cfg)
    lines = [f"# {cfg.report.title}", ""]
    if meta.get("finished_at") or meta.get("started_at"):
        lines += [f"Run `{meta.get('run_id', '')}`, {meta.get('finished_at') or meta.get('started_at')}.", ""]
    if not summaries:
        lines += ["No model completed a test.", ""]
        return "\n".join(lines)

    header = ["Rank", "Model", "Overall", "Passed", "Accuracy", *[_title(c) for c in categories],
              "TTFT (s)", "Tokens/Sec", "Avg Time (s)", "VRAM / GPU @ Ctx", "Prompt Eval"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| :--- | :--- | " + " | ".join([":---:"] * (len(header) - 2)) + " |")
    for rank, s in enumerate(summaries, start=1):
        passed = f"{s.passed}/{s.with_mark}" if s.with_mark else "-"
        cats = [f"{s.categories[c]}%" if c in s.categories else "-" for c in categories]
        row = [f"#{rank}", f"**{_cell(s.model)}**", f"**{s.overall}**", passed, f"{s.accuracy}%", *cats,
               f"{s.ttft_s}s", f"**{s.tokens_per_sec} tok/s**", f"{s.avg_time_s}s", _cell(s.vram_label),
               f"{s.prompt_eval_tps} tok/s"]
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", CATEGORY_NOTE]
    used = {t.difficulty for s in summaries for t in s.tests}
    factors = {d: cfg.scoring.difficulty_weights.get(d, 1.0) for d in ("easy", "medium", "hard", "expert") if d in used}
    if len(set(factors.values())) > 1:
        lines += ["", "Harder tests count for more in Accuracy and the category columns: "
                  + ", ".join(f"{d} x{w:g}" for d, w in factors.items()) + "."]

    unused = unused_categories(summaries, cfg)
    if unused:
        names = " and ".join(_title(c).lower() for c in unused)
        lines += ["", f"This run has no {names} tests, so Overall leaves that part out and scales the other weights "
                  "up to 100%. Compare Overall only between runs of the same tests."]

    not_counted = skipped_tests(records)
    if not_counted:
        names = {r.test_id: r.test_name for r in records}
        lines += ["", "Not counted in the scores above, because not every model could run them (models are compared "
                  "on the same tests): "
                  + "; ".join(f"**{_cell(names[t])}** (skipped by {', '.join(models)})" for t, models in not_counted.items())
                  + ". Their results are under Results by test."]

    best = summaries[0]
    lines += ["", "## Category winners", f"- **Best overall:** `{best.model}` ({best.overall} score)"]
    for category in categories:
        scored = [s for s in summaries if category in s.categories]
        if scored:
            top = max(scored, key=lambda s: (s.categories[category], _passed_in(s, category)[0], s.tokens_per_sec))
            passed, marked = _passed_in(top, category)
            tally = f"{passed}/{marked} passed, " if marked else ""
            lines.append(f"- **Best {_title(category).lower()}:** `{top.model}` ({top.categories[category]}%, {tally}"
                         f"{top.tokens_per_sec} tok/s)")
    marked = [s for s in summaries if s.with_mark]
    if marked:
        top = max(marked, key=lambda s: (s.pass_rate, s.overall))
        lines.append(f"- **Most tests passed:** `{top.model}` ({top.passed}/{top.with_mark})")
    fastest = max(summaries, key=lambda s: s.tokens_per_sec)
    snappiest = min(summaries, key=lambda s: s.ttft_s)
    lines.append(f"- **Fastest generation:** `{fastest.model}` ({fastest.tokens_per_sec} tok/s)")
    lines.append(f"- **Lowest TTFT:** `{snappiest.model}` ({snappiest.ttft_s}s)")

    lines += ["", "## Results by test", ""]
    by_test: OrderedDict[str, list[AttemptRecord]] = OrderedDict()
    for record in records:
        by_test.setdefault(record.test_id, []).append(record)
    goals = meta.get("tests", {})
    for test_id, rows in by_test.items():
        first = rows[0]
        info = goals.get(test_id, {})
        lines.append(f"### {first.test_name}")
        lines.append("")
        if info.get("description"):
            lines.append(f"- **Asked:** {info['description']}")
        if info.get("goal"):
            lines.append(f"- **Goal:** {info['goal']}")
        lines.append(f"- **Pass mark:** {first.pass_mark or 'none (measured only)'}")
        lines += ["", "| Model | Result | Accuracy | Quality | Missed expectations |", "| :--- | :---: | :---: | :---: | :--- |"]
        for row in rows:
            if row.skipped is not None:
                lines.append(f"| {_cell(row.model)} | SKIPPED | - | - | {_cell(row.skipped)} |")
                continue
            if not row.ok:
                result, missed = "ERROR", _cell(row.error or "")
            else:
                result = "PASS" if row.passed else ("FAIL" if row.passed is False else "measured")
                misses = [c.expect for c in row.checks if not c.met and not c.skipped]
                missed = _cell("; ".join(misses)) if misses else "none"
                if not row.response.strip():
                    missed = _cell("No answer. " + " ".join(row.warnings)).strip()
                elif row.warnings:
                    missed += " (" + _cell(" ".join(row.warnings)) + ")"
            label = row.model + (f" (repeat {row.repeat})" if row.repeat > 1 or len(rows) > len({r.model for r in rows}) else "")
            lines.append(f"| {_cell(label)} | {result} | {row.accuracy} | {row.quality} | {missed} |")
        lines.append("")
    return "\n".join(lines)
