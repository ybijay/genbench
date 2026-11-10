"""Reports built from results.jsonl: leaderboard.md, results.csv, summary.json."""

from __future__ import annotations

from pathlib import Path

from genbench.config import Config
from genbench.report.markdown import leaderboard_markdown
from genbench.report.tabular import write_csv, write_summary_json
from genbench.results import AttemptRecord
from genbench.score import ModelSummary, summarize


def write_reports(run_dir: Path, records: list[AttemptRecord], cfg: Config, meta: dict | None = None) -> tuple[list[ModelSummary], list[Path]]:
    summaries = summarize(records, cfg)
    written: list[Path] = []
    formats = set(cfg.report.formats)
    if "markdown" in formats:
        path = run_dir / "leaderboard.md"
        path.write_text(leaderboard_markdown(summaries, records, cfg, meta or {}), encoding="utf-8")
        written.append(path)
    if "csv" in formats:
        written.append(write_csv(run_dir / "results.csv", records))
    if "json" in formats or "jsonl" in formats:
        written.append(write_summary_json(run_dir / "summary.json", summaries, meta or {}, records))
    return summaries, written
