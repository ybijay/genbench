"""results.csv (the original script's columns first, so existing spreadsheets keep working) and summary.json."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from genbench.results import AttemptRecord, skipped_tests
from genbench.score import ModelSummary

LEGACY_COLUMNS = [
    "Model", "Task", "Category", "Response Time (s)", "Warm TTFT (s)", "Tokens/Sec", "Prompt Eval (tok/s)",
    "Prompt Tokens", "Completion Tokens", "Accuracy (0-100)", "Category Score (0-100)", "VRAM Allocation",
    "Cold Load (s)", "Verification Notes", "Output",
]
EXTRA_COLUMNS = ["Test ID", "Result", "Pass Mark", "Repeat", "Wall TTFT (s)", "Thinking (chars)", "Error",
                 "Quality (0-100)"]


def write_csv(path: Path, records: list[AttemptRecord]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LEGACY_COLUMNS + EXTRA_COLUMNS)
        writer.writeheader()
        for r in records:
            result = "ERROR" if not r.ok else ("PASS" if r.passed else ("FAIL" if r.passed is False else "measured"))
            if r.skipped is not None:
                result = "SKIPPED"
            trouble = "SKIPPED" if r.skipped is not None else "ERROR"
            writer.writerow({
                "Model": r.model,
                "Task": r.test_name,
                "Category": r.category,
                "Response Time (s)": r.total_s if r.ok else trouble,
                "Warm TTFT (s)": r.warm_ttft_s if r.ok else 0.0,
                "Tokens/Sec": r.tokens_per_sec if r.ok else 0.0,
                "Prompt Eval (tok/s)": r.prompt_eval_tps if r.ok else 0.0,
                "Prompt Tokens": r.prompt_tokens,
                "Completion Tokens": r.completion_tokens,
                "Accuracy (0-100)": r.accuracy,
                "Category Score (0-100)": r.category_points,
                "VRAM Allocation": r.vram_label,
                "Cold Load (s)": r.cold_load_s,
                "Verification Notes": r.notes if r.ok else (f"Skipped: {r.skipped}" if r.skipped else f"Error: {r.error}"),
                "Output": r.response if r.ok else (r.error or ""),
                "Test ID": r.test_id,
                "Result": result,
                "Pass Mark": r.pass_mark or "",
                "Repeat": r.repeat,
                "Wall TTFT (s)": r.wall_ttft_s,
                "Thinking (chars)": len(r.thinking),
                "Error": r.error or "",
                "Quality (0-100)": r.quality,
            })
    return path


def write_summary_json(path: Path, summaries: list[ModelSummary], meta: dict,
                       records: list[AttemptRecord] | None = None) -> Path:
    data = {
        "run_id": meta.get("run_id"),
        "not_counted": skipped_tests(records or []),
        "models": [{**asdict(s), "pass_rate": s.pass_rate, "rank": i} for i, s in enumerate(summaries, start=1)],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
