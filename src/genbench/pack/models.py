"""The test-pack format. A pack is a folder with a pack.json; each test in it explains itself."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from genbench.config import PassMark
from genbench.context import ContextSpec
from genbench.pack.checks import Check, visit

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Slug = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]*$")]
Suite = Literal["quick", "standard", "full", "perf"]
Difficulty = Literal["easy", "medium", "hard", "expert"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Extract(_Strict):
    kind: Literal["code_block", "json_block", "final_answer", "tool_calls", "whole"] = "whole"
    langs: list[str] = Field(default_factory=list, description="Fence languages to look for first, e.g. ['python', 'py']")
    fallback_langs: list[str] | None = Field(
        None, description="Fence languages to try when none of 'langs' match; '' means an untagged fence"
    )
    pick: Literal["longest", "first", "last"] = "longest"
    on_no_block: Literal["whole_response", "fail"] = "whole_response"
    save_as: str | None = Field(None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$", description="File name in the run folder")
    strip_thinking: bool = True


class Message(_Strict):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class Rubric(_Strict):
    accuracy: list[Check] = Field(min_length=1, description="Did the answer do what was asked?")
    quality: list[Check] | Literal["same_as_accuracy"] = Field(
        "same_as_accuracy", description="How well it was done (format, polish, depth)"
    )
    category_score: Literal["mean", "quality", "accuracy"] = Field(
        "mean",
        description="What this test adds to its category column: the average of accuracy and quality (mean), "
        "or one of them alone. The core pack uses 'quality', as the original script did",
    )

    @model_validator(mode="after")
    def _expect_on_top_level(self) -> Rubric:
        for section, checks in self.sections():
            for i, check in enumerate(checks):
                if not check.expect:
                    raise ValueError(
                        f"{section}[{i}] ({check.check}) needs an 'expect' line: the plain-English "
                        "expectation that reports show as met or missed"
                    )
        return self

    def sections(self) -> list[tuple[str, list[Any]]]:
        out: list[tuple[str, list[Any]]] = [("accuracy", list(self.accuracy))]
        if isinstance(self.quality, list):
            out.append(("quality", list(self.quality)))
        return out

    def max_score(self, section: str) -> float:
        checks = dict(self.sections()).get(section, [])
        return sum(check.max_score() for check in checks)


class BenchTest(_Strict):
    __test__ = False  # keep pytest from collecting this class

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    name: Text
    description: Text = Field(description="What the model is asked to do")
    goal: Text = Field(description="What a passing answer looks like")
    pass_: PassMark | None = Field(
        None, alias="pass", description="Pass mark; omit to use [scoring].pass_mark, null for measured-only tests"
    )
    category: Slug
    difficulty: Difficulty = "medium"
    tags: list[str] = Field(default_factory=list)
    suites: list[Suite] = Field(default_factory=lambda: ["standard", "full"])
    aliases: list[str] = Field(default_factory=list, description="Other names, e.g. the task name in old result files")
    prompt: str | list[str] | None = Field(None, description="The prompt; a list of strings is joined with newlines")
    prompt_file: str | None = Field(None, description="Prompt file, relative to the pack folder")
    messages: list[Message] | None = None
    system: str | None = None
    vars: dict[str, Any] = Field(default_factory=dict)
    context: ContextSpec | None = Field(None, description="Generated text that replaces {{context}} in the prompt")
    tools: list[dict[str, Any]] | None = None
    format: dict[str, Any] | Literal["json"] | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    timeout_s: float | None = Field(None, gt=0)
    repeats: int | None = Field(None, ge=1)
    extract: Extract = Field(default_factory=Extract)
    rubric: Rubric

    @model_validator(mode="after")
    def _one_prompt_source(self) -> BenchTest:
        given = [name for name in ("prompt", "prompt_file", "messages") if getattr(self, name) is not None]
        if len(given) != 1:
            found = f" (found {', '.join(given)})" if given else ""
            raise ValueError(f"set exactly one of 'prompt', 'prompt_file' or 'messages'{found}")
        return self

    def pass_mark(self, default: PassMark) -> PassMark | None:
        """The test's own mark, the config default when it sets none, or None when it is measured only."""
        return self.pass_ if "pass_" in self.model_fields_set else default

    def prompt_text(self, pack_dir: Path) -> str | None:
        if self.prompt is not None:
            return "\n".join(self.prompt) if isinstance(self.prompt, list) else self.prompt
        if self.prompt_file is not None:
            text = (pack_dir / self.prompt_file).read_text(encoding="utf-8-sig")
            return text.replace("\r\n", "\n").rstrip("\n")
        return None

    def walk_checks(self) -> Iterator[tuple[str, str, Any, int, str | None]]:
        """(section, path, check, depth, parent kind) for every check, nested ones included, in order."""
        for section, checks in self.rubric.sections():
            for i, check in enumerate(checks):
                for path, item, depth, parent in visit(f"rubric.{section}[{i}]", check):
                    yield section, path, item, depth, parent


class Pack(_Strict):
    schema_ref: str | None = Field(None, alias="$schema")
    pack: Slug
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    description: str = ""
    tests: list[BenchTest] = Field(min_length=1)
