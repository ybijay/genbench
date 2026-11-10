"""Run syntax checkers on generated code. Nothing here ever executes the code itself."""

from __future__ import annotations

import ast
import json
import subprocess
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from genbench.config import ValidatorSettings

EXTENSIONS = {
    "python": ".py", "ruff": ".py", "bash": ".sh", "shellcheck": ".sh", "rustfmt": ".rs",
    "tsc": ".ts", "sqlite3": ".sql", "sqlite": ".sql", "powershell": ".ps1", "pwsh": ".ps1",
}


@dataclass(frozen=True)
class Outcome:
    status: Literal["ok", "fail", "missing"]
    detail: str
    findings: int = 0


class ValidatorRunner:
    """force_fallback: True for every tool, or a set of tool names, to act as if those tools were missing."""

    def __init__(self, settings: ValidatorSettings, force_fallback: bool | Collection[str] = ()):
        self.settings = settings
        self.force_fallback = force_fallback

    @property
    def on_missing(self) -> str:
        return self.settings.on_missing

    def _forced(self, name: str) -> bool:
        return self.force_fallback is True or (not isinstance(self.force_fallback, bool) and name in self.force_fallback)

    def run(self, name: str, code: str, workdir: Path, filename: str | None = None) -> Outcome:
        tool = self.settings.tools.get(name)
        if tool is None:
            return Outcome("missing", f"{name}: not configured in [validators]")
        if self._forced(name):
            return Outcome("missing", f"{name}: fallback forced")
        if tool.builtin == "ast":
            try:
                ast.parse(code)
            except SyntaxError as exc:
                return Outcome("fail", f"{name}: SyntaxError: {exc.msg} (line {exc.lineno})", 1)
            except (ValueError, MemoryError, RecursionError) as exc:
                return Outcome("fail", f"{name}: {type(exc).__name__}: {exc}", 1)
            return Outcome("ok", f"{name}: ast.parse OK")

        assert tool.cmd is not None
        workdir.mkdir(parents=True, exist_ok=True)
        path = workdir / (filename or f"validate_{name}{EXTENSIONS.get(name, '.txt')}")
        path.write_text(code + "\n", encoding="utf-8", newline="\n")
        uses_file = any("{file}" in arg for arg in tool.cmd)
        args = [arg.replace("{file}", path.as_posix()) for arg in tool.cmd]
        try:
            proc = subprocess.run(
                args,
                input=None if uses_file else code,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=tool.timeout_s,
                cwd=workdir,
            )
        except (FileNotFoundError, PermissionError, NotADirectoryError, OSError):
            return Outcome("missing", f"{name}: {tool.cmd[0]} not available")
        except subprocess.TimeoutExpired:
            return Outcome("missing", f"{name}: timed out after {tool.timeout_s:g}s")
        findings = count_findings(proc.stdout)
        if proc.returncode == 0:
            return Outcome("ok", f"{name}: OK", findings)
        first = next((line.strip() for line in (proc.stderr + "\n" + proc.stdout).splitlines() if line.strip()), "")
        return Outcome("fail", f"{name}: FAIL" + (f" ({first[:120]})" if first else ""), max(findings, 1))


def count_findings(stdout: str) -> int:
    """A JSON list (ruff, shellcheck -f json) counts its items; otherwise non-empty output lines."""
    text = stdout.strip()
    if not text:
        return 0
    try:
        data = json.loads(text)
    except ValueError:
        return sum(1 for line in text.splitlines() if line.strip())
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                return len(value)
    return 0
