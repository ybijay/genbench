"""Re-grade results saved by the original ollama_local_benchmark.py and compare scores (the Phase 1 gate)."""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

from genbench.config import Config
from genbench.grade.engine import GradeContext, grade_test
from genbench.grade.validators import ValidatorRunner
from genbench.pack.loader import Catalog
from genbench.results import AttemptRecord, artifact_dir, slug, test_hash

REQUIRED = {"Model", "Task", "Category", "Response Time (s)", "Accuracy (0-100)", "Category Score (0-100)", "Output"}
HEURISTIC_NOTES = ("Bash Structure", "Incomplete Bash blocks")


@dataclass
class ParityRow:
    model: str
    test_id: str
    old_accuracy: float
    new_accuracy: float
    old_category: float
    new_category: float

    @property
    def worst_gap(self) -> float:
        return max(abs(self.new_accuracy - self.old_accuracy), abs(self.new_category - self.old_category))


def _float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except ValueError:
        return default


def read_legacy_csv(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} doesn't look like an ollama_local_benchmark.py results file (missing {', '.join(sorted(missing))})")
        return list(reader)


def regrade_legacy(
    path: Path, catalog: Catalog, cfg: Config, run_dir: Path, validator_mode: str = "auto"
) -> tuple[list[AttemptRecord], list[ParityRow]]:
    """validator_mode: auto (use the old heuristic where the saved notes show it was used), tool or fallback."""
    rows = read_legacy_csv(path)
    run_id = run_dir.name
    records: list[AttemptRecord] = []
    parity: list[ParityRow] = []
    for row in rows:
        found = catalog.find(row["Task"])
        if found is None:
            raise ValueError(f"no test matches the saved task '{row['Task']}' (add it to a test's aliases)")
        test = found.test
        notes = row.get("Verification Notes", "")
        record = AttemptRecord(
            run_id=run_id, model=row["Model"], endpoint="legacy", test_id=test.id, test_name=test.name,
            pack=found.pack.pack.pack, pack_version=found.pack.pack.version, test_hash=test_hash(found),
            category=test.category, difficulty=test.difficulty, vram_label=row.get("VRAM Allocation") or "N/A",
            cold_load_s=_float(row.get("Cold Load (s)")), response=row["Output"],
            legacy={"accuracy": _float(row["Accuracy (0-100)"]), "category_score": _float(row["Category Score (0-100)"]),
                    "notes": notes, "task": row["Task"]},
        )
        if row["Response Time (s)"] == "ERROR":
            record.error = notes or "error in the original run"
            records.append(record)
            continue
        record.total_s = _float(row["Response Time (s)"])
        record.warm_ttft_s = _float(row.get("Warm TTFT (s)"))
        record.wall_ttft_s = record.warm_ttft_s
        record.tokens_per_sec = _float(row.get("Tokens/Sec"))
        record.prompt_eval_tps = _float(row.get("Prompt Eval (tok/s)"))
        record.prompt_tokens = int(_float(row.get("Prompt Tokens")))
        record.completion_tokens = int(_float(row.get("Completion Tokens")))

        if validator_mode == "fallback":
            force: bool | set[str] = True
        elif validator_mode == "tool":
            force = set()
        else:
            force = {"bash"} if any(marker in notes for marker in HEURISTIC_NOTES) else set()
        ctx = GradeContext(
            validators=ValidatorRunner(cfg.validators, force),
            workdir=artifact_dir(run_dir, f"{slug(record.model)}", test.id, 1, 1),
            pack_dir=found.pack.dir,
            default_pass_mark=cfg.scoring.pass_mark,
            model=record.model,
        )
        record.apply_grade(grade_test(test, record.response, ctx), run_dir)
        records.append(record)
        parity.append(ParityRow(record.model, test.id, record.legacy["accuracy"], record.accuracy,
                                record.legacy["category_score"], record.quality))
    return records, parity
