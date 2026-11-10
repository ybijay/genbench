"""Check types used in test rubrics.

A check awards `points` when it passes and `fail_points` (default 0) when it doesn't. Graded checks
(word_count, code_ratio, tool_calls, validator_findings, external, scaled_score) award anything up to
their maximum. Top-level checks carry an `expect` line: the plain-English expectation that reports show
as met or missed. Phase 0 defines and validates the format; evaluation arrives in Phase 1.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

COMPOSITES = ("all_of", "any_of", "none_of")
GRADED = ("word_count", "code_ratio", "tool_calls", "validator_findings", "external", "scaled_score")


class _CheckBase(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str | None = Field(None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$", description="Name for 'passed' checks to refer to")
    expect: str | None = Field(
        None, min_length=1, description="Plain-English expectation shown as met or missed; required on top-level checks"
    )
    points: float = Field(0, ge=0, description="Points when the check passes")
    fail_points: float = Field(0, ge=0, description="Points when the check fails (default 0)")
    on: Literal["extracted", "response", "thinking"] = Field(
        "extracted", description="What the check reads: the extracted artifact, the whole response, or the thinking text"
    )

    def max_score(self) -> float:
        return max(self.points, self.fail_points)

    def children(self) -> list[Any]:
        return []


class Contains(_CheckBase):
    check: Literal["contains"]
    value: str = Field(min_length=1)
    case_sensitive: bool = True


class NotContains(_CheckBase):
    check: Literal["not_contains"]
    value: str = Field(min_length=1)
    case_sensitive: bool = True


class Regex(_CheckBase):
    check: Literal["regex"]
    pattern: str = Field(min_length=1)
    flags: str = Field("", pattern=r"^[imsx]*$", description="i ignore case, m multiline, s dot matches newline, x verbose")
    min_matches: int = Field(1, ge=1)
    engine: Literal["re", "regex"] = "re"


class NotRegex(_CheckBase):
    check: Literal["not_regex"]
    pattern: str = Field(min_length=1)
    flags: str = Field("", pattern=r"^[imsx]*$")
    engine: Literal["re", "regex"] = "re"


class AllOf(_CheckBase):
    check: Literal["all_of"]
    of: list[Check] = Field(min_length=1)

    def children(self) -> list[Any]:
        return list(self.of)


class AnyOf(_CheckBase):
    check: Literal["any_of"]
    of: list[Check] = Field(min_length=1)

    def children(self) -> list[Any]:
        return list(self.of)


class NoneOf(_CheckBase):
    check: Literal["none_of"]
    of: list[Check] = Field(min_length=1)

    def children(self) -> list[Any]:
        return list(self.of)


class Passed(_CheckBase):
    check: Literal["passed"]
    ref: str = Field(min_length=1, description="id of an earlier check")


class WordTier(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_words: int = Field(ge=0)
    points: float = Field(ge=0)


class WordCount(_CheckBase):
    check: Literal["word_count"]
    tiers: list[WordTier] = Field(min_length=1, description="The highest tier reached wins")

    def max_score(self) -> float:
        return max(t.points for t in self.tiers)


class CodeRatio(_CheckBase):
    check: Literal["code_ratio"]
    scale: float = Field(gt=0, description="points = min(max_points, ratio * scale); ratio = extracted / response length")
    max_points: float = Field(gt=0)

    def max_score(self) -> float:
        return self.max_points


class Detect(_CheckBase):
    check: Literal["detect"]
    what: Literal["markdown_table", "ascii_diagram", "shebang", "fenced_block"]


class JsonValid(_CheckBase):
    check: Literal["json_valid"]


class JsonSchema(_CheckBase):
    check: Literal["json_schema"]
    schema_: dict[str, Any] | None = Field(None, alias="schema")
    schema_file: str | None = None

    @model_validator(mode="after")
    def _one_source(self) -> JsonSchema:
        if (self.schema_ is None) == (self.schema_file is None):
            raise ValueError("set exactly one of 'schema' or 'schema_file'")
        return self


class YamlValid(_CheckBase):
    check: Literal["yaml_valid"]


class ExpectedCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict, description="Arguments that must match exactly; others are ignored")
    points: float = Field(ge=0)


class ToolCalls(_CheckBase):
    check: Literal["tool_calls"]
    source: Literal["json_in_text", "native"] = "json_in_text"
    calls: list[ExpectedCall] = Field(min_length=1)
    max_points: float | None = Field(None, gt=0)
    count_duplicates: bool = True

    def max_score(self) -> float:
        return self.max_points if self.max_points is not None else sum(c.points for c in self.calls)


class NumericAnswer(_CheckBase):
    check: Literal["numeric_answer"]
    expected: float
    tolerance: float = Field(0, ge=0)
    relative: bool = False
    pattern: str | None = Field(None, description="Regex whose first group captures the number; default: the last number")


class ChoiceAnswer(_CheckBase):
    check: Literal["choice_answer"]
    expected: str = Field(min_length=1)
    pattern: str = r"ANSWER:\s*([A-Za-z0-9]+)"
    case_sensitive: bool = False


class Constraint(_CheckBase):
    check: Literal["constraint"]
    min_words: int | None = Field(None, ge=0)
    max_words: int | None = Field(None, ge=0)
    bullet_count: int | None = Field(None, ge=0)
    max_words_per_bullet: int | None = Field(None, ge=1)
    banned_words: list[str] = Field(default_factory=list)
    must_end_with: str | None = None

    @model_validator(mode="after")
    def _something_to_check(self) -> Constraint:
        fields = (self.min_words, self.max_words, self.bullet_count, self.max_words_per_bullet, self.must_end_with)
        if all(v is None for v in fields) and not self.banned_words:
            raise ValueError("set at least one constraint")
        return self


class Validator(_CheckBase):
    check: Literal["validator"]
    tool: str = Field(min_length=1, description="A tool from [validators] in genbench.toml")
    fallback: Check | None = Field(
        None,
        description="Used when the tool isn't available: passing earns this check's points, "
        "failing earns the fallback's fail_points",
    )

    def children(self) -> list[Any]:
        return [self.fallback] if self.fallback is not None else []


class ValidatorFindings(_CheckBase):
    check: Literal["validator_findings"]
    tool: str = Field(min_length=1)
    max_points: float = Field(gt=0)
    penalty_per_finding: float = Field(gt=0)

    def max_score(self) -> float:
        return self.max_points


class External(_CheckBase):
    check: Literal["external"]
    cmd: list[str] = Field(min_length=1, description="Command; it reads a JSON request on stdin and prints {score, notes}")
    timeout_s: float = Field(30, gt=0)
    max_points: float = Field(gt=0)

    def max_score(self) -> float:
        return self.max_points


class ScaledScore(_CheckBase):
    check: Literal["scaled_score"]
    of_rubric: Literal["accuracy"] = "accuracy"
    factor: float = Field(gt=0)
    max_points: float = Field(gt=0)

    def max_score(self) -> float:
        return self.max_points


class ForbiddenCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict, description="Arguments that must all match; {} matches any call to the tool")


class NoToolCalls(_CheckBase):
    check: Literal["no_tool_calls"]
    source: Literal["json_in_text", "native"] = "json_in_text"
    calls: list[ForbiddenCall] = Field(
        min_length=1, description="Calls that must not appear. Fails too when no tool-call list is found at all"
    )


class TableRow(_CheckBase):
    check: Literal["table_row"]
    match: list[str] = Field(min_length=1, description="Regexes (case-insensitive); some markdown table row must match all")


class NoTableRow(_CheckBase):
    check: Literal["no_table_row"]
    match: list[str] = Field(
        min_length=1, description="Regexes (case-insensitive); no table row may match all. Fails when there is no table"
    )


class TableRows(_CheckBase):
    check: Literal["table_rows"]
    min_rows: int = Field(1, ge=0)
    max_rows: int | None = Field(None, ge=0)
    all_match: list[str] = Field(default_factory=list, description="Regexes (case-insensitive) every row must match")


Check = Annotated[
    Contains | NotContains | Regex | NotRegex | AllOf | AnyOf | NoneOf | Passed | WordCount | CodeRatio | Detect | JsonValid | JsonSchema | YamlValid | ToolCalls | NumericAnswer | ChoiceAnswer | Constraint | Validator | ValidatorFindings | External | ScaledScore | NoToolCalls | TableRow | NoTableRow | TableRows,
    Field(discriminator="check"),
]

for _model in (AllOf, AnyOf, NoneOf, Validator):
    _model.model_rebuild()

CHECK_TYPES = sorted(
    cls.model_fields["check"].annotation.__args__[0]  # type: ignore[union-attr]
    for cls in (
        Contains, NotContains, Regex, NotRegex, AllOf, AnyOf, NoneOf, Passed, WordCount, CodeRatio, Detect,
        JsonValid, JsonSchema, YamlValid, ToolCalls, NumericAnswer, ChoiceAnswer, Constraint, Validator,
        ValidatorFindings, External, ScaledScore, NoToolCalls, TableRow, NoTableRow, TableRows,
    )
)


def visit(path: str, check: Any, depth: int = 0, parent: str | None = None) -> Iterator[tuple[str, Any, int, str | None]]:
    """Yield (path, check, depth, parent kind) for a check and everything nested in it, in order."""
    yield path, check, depth, parent
    if check.check in COMPOSITES:
        for i, child in enumerate(check.of):
            yield from visit(f"{path}.of[{i}]", child, depth + 1, check.check)
    elif check.check == "validator" and check.fallback is not None:
        yield from visit(f"{path}.fallback", check.fallback, depth + 1, "fallback")
