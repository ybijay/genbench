"""genbench command line: init, validate, explain, list, new-test, schema, run, regrade, report."""

import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import typer
from pydantic import ValidationError
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from genbench import __version__
from genbench.config import (
    CONFIG_FILENAME,
    ENV_CONFIG,
    Config,
    ConfigError,
    find_config,
    format_validation_error,
    load_config,
)
from genbench.pack.checks import COMPOSITES, GRADED
from genbench.pack.loader import Issue, LoadedTest, load_catalog, select_tests
from genbench.pack.validate import validate_catalog

app = typer.Typer(
    name="genbench",
    help="Configurable local LLM benchmark. Settings come from genbench.toml; tests come from JSON packs.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
console = Console(highlight=False)
err_console = Console(stderr=True, highlight=False)

TEST_ID_HELP = "Lowercase letters, digits and underscores, e.g. sql_window_analytics"


@dataclass
class _State:
    config: Path | None = None
    overrides: list[str] = field(default_factory=list)


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"genbench {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    config: Path | None = typer.Option(
        None, "--config", "-c", help=f"Config file (default: ${ENV_CONFIG}, then ./{CONFIG_FILENAME})."
    ),
    set_: list[str] = typer.Option(
        [], "--set", metavar="KEY=VALUE", help="Override one setting, e.g. --set run.repeats=3. Repeatable."
    ),
    version: bool = typer.Option(False, "--version", callback=_show_version, is_eager=True, help="Show the version."),
) -> None:
    ctx.obj = _State(config, list(set_))


def _say(message: str, style: str | None = None) -> None:
    console.print(Text(message, style=style or ""))


def _fail(message: str, code: int = 1) -> typer.Exit:
    err_console.print(Text(message, style="red"))
    return typer.Exit(code)


def _load(ctx: typer.Context) -> Config:
    state: _State = ctx.obj or _State()
    try:
        return load_config(find_config(state.config), state.overrides)
    except ConfigError as exc:
        raise _fail(f"Config error: {exc}", 2) from None


# --------------------------------------------------------------------------- validate


@app.command()
def validate(
    ctx: typer.Context,
    portable: bool = typer.Option(False, "--portable", help="Also warn about regex features Rust v2 won't support."),
    online: bool = typer.Option(False, "--online", help="Also contact each endpoint and check the models exist."),
    strict: bool = typer.Option(False, "--strict", help="Fail on warnings too."),
) -> None:
    """Check genbench.toml and every test pack without running a model."""
    cfg = _load(ctx)
    _print_setup(cfg)
    catalog = load_catalog(cfg)
    issues = validate_catalog(catalog, cfg, portable=portable)
    if online:
        issues.extend(_online_issues(cfg))
    for issue in issues:
        console.print(_issue_text(issue))
    errors = sum(1 for i in issues if i.level == "error")
    warnings = len(issues) - errors
    _say(f"{len(catalog.packs)} pack(s), {len(catalog.tests)} test(s): {errors} error(s), {warnings} warning(s).")
    if errors or (strict and warnings):
        raise typer.Exit(1)
    _say("Validation passed.", "green")


def _print_setup(cfg: Config) -> None:
    source = str(cfg.source) if cfg.source else f"none found, using defaults (no ./{CONFIG_FILENAME}, ${ENV_CONFIG} not set)"
    _say(f"Config:   {source}")
    for ep in cfg.endpoints:
        fallbacks = f" (fallback {', '.join(ep.fallback_urls)})" if ep.fallback_urls else ""
        _say(f"Endpoint: {ep.name} [{ep.kind}] {ep.url}{fallbacks}")
    enabled = cfg.enabled_models()
    of_total = f" of {len(cfg.models)}" if len(enabled) != len(cfg.models) else ""
    if cfg.models:
        _say(f"Models:   {len(enabled)}{of_total} enabled")
    else:
        _say("Models:   none listed ('genbench run' offers the ones installed on the server)")


def _issue_text(issue: Issue) -> Text:
    text = Text()
    text.append(f"{issue.level.upper():<8}", style="red" if issue.level == "error" else "yellow")
    text.append(f"{issue.where}\n        {issue.message}")
    return text


def _online_issues(cfg: Config) -> list[Issue]:
    from genbench.backends.probe import has_model, probe

    out: list[Issue] = []
    for ep in cfg.endpoints:
        answer = probe(ep)
        where = f"endpoint {ep.name}"
        if not answer.ok:
            out.append(Issue("error", where, f"not reachable: {answer.error}"))
            continue
        _say(f"Reached {ep.name} at {answer.url}: {len(answer.models)} model(s) available.")
        for model in cfg.enabled_models():
            if (model.endpoint or cfg.endpoints[0].name) == ep.name and not has_model(answer.models, model.name):
                out.append(Issue("error", where, f"model '{model.name}' isn't on this server"))
    return out


# --------------------------------------------------------------------------- explain


@app.command()
def explain(
    ctx: typer.Context,
    test_id: str = typer.Argument(..., help="Test id or alias (see 'genbench list tests')."),
    full_prompt: bool = typer.Option(False, "--full-prompt", help="Print the whole prompt."),
) -> None:
    """Show what a test asks, what passing means and how each expectation is checked."""
    cfg = _load(ctx)
    found = load_catalog(cfg).find(test_id)
    if found is None:
        raise _fail(f"No test '{test_id}'. Run 'genbench list tests' to see the ids.")
    test = found.test
    mark = test.pass_mark(cfg.scoring.pass_mark)
    if mark is None:
        mark_text = "none: this test is measured, not passed"
    else:
        origin = "set by the test" if "pass_" in test.model_fields_set else "default from [scoring].pass_mark"
        mark_text = f"{mark.describe()} ({origin})"

    _say(test.name, "bold")
    _say(
        f"id {test.id} | pack {found.pack.label} | {test.category} | {test.difficulty} | "
        f"suites: {', '.join(test.suites) or 'none'}" + (f" | tags: {', '.join(test.tags)}" if test.tags else "")
    )
    _say("")
    _field("Asked", test.description)
    _field("Goal", test.goal)
    _field("Pass mark", mark_text)
    adds = {"mean": "the average of accuracy and quality", "quality": "its quality score",
            "accuracy": "its accuracy score"}[test.rubric.category_score]
    _field("Counts as", f"{adds}, in the {test.category.replace('_', ' ')} column")
    if test.context is not None:
        from genbench.context import describe

        _field("Context", describe(test.context))
        from genbench.runner import build_messages, context_needed

        _field("Needs", f"a context window of about {context_needed(build_messages(found)):,} tokens; "
               "models with less skip the test, and it is then left out of every model's scores")
    for section, checks in test.rubric.sections():
        _say("")
        _say(f"{section.capitalize()} expectations ({test.rubric.max_score(section):g} points)", "bold")
        table = Table(box=None, show_header=True, pad_edge=False, padding=(0, 2, 0, 0))
        table.add_column("Points", justify="right", no_wrap=True)
        table.add_column("Expectation")
        table.add_column("Checked by", style="dim")
        for check in checks:
            table.add_row(Text(_points(check)), Text(check.expect or ""), Text(_summary(check)))
        console.print(table)
    if test.rubric.quality == "same_as_accuracy":
        _say("")
        _say("Quality: same as accuracy")
    prompt = found.prompt_text() or ""
    if full_prompt and test.context is not None:
        from genbench.runner import prompt_variables, render_prompt

        prompt = render_prompt(prompt, prompt_variables(found))
    if test.messages is not None:
        prompt = "\n".join(f"[{m.role}] {m.content}" for m in test.messages)
    source = test.prompt_file or ("messages" if test.messages is not None else "inline")
    _say("")
    _say(f"Prompt ({source}, {len(prompt)} characters)", "bold")
    lines = prompt.splitlines()
    shown = lines if full_prompt else lines[:8]
    for line in shown:
        _say(f"  {line}")
    if len(shown) < len(lines):
        _say(f"  ... {len(lines) - len(shown)} more line(s); use --full-prompt", "dim")


def _field(label: str, value: str) -> None:
    text = Text()
    text.append(f"{label + ':':<11}", style="bold")
    text.append(value)
    console.print(text)


def _points(check: object) -> str:
    kind = check.check
    if kind in GRADED:
        return f"up to {check.max_score():g}"  # type: ignore[attr-defined]
    points = f"{check.points:g}"  # type: ignore[attr-defined]
    fail = getattr(check, "fail_points", 0)
    return f"{points} (else {fail:g})" if fail else points


def _short(text: str, limit: int = 48) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _summary(check: object) -> str:
    c = check  # pydantic model; attributes depend on the kind
    kind = c.check  # type: ignore[attr-defined]
    label = kind.replace("_", " ")
    if kind in ("contains", "not_contains"):
        text = f'{label} "{_short(c.value)}"' + ("" if c.case_sensitive else " (any case)")  # type: ignore[attr-defined]
    elif kind in ("regex", "not_regex"):
        text = f"{label} /{_short(c.pattern)}/{c.flags}"  # type: ignore[attr-defined]
    elif kind in COMPOSITES:
        text = f"{label} {len(c.of)} checks"  # type: ignore[attr-defined]
    elif kind == "passed":
        text = f"passed: {c.ref}"  # type: ignore[attr-defined]
    elif kind == "validator":
        text = f"validator {c.tool}" + (" (fallback set)" if c.fallback else "")  # type: ignore[attr-defined]
    elif kind == "validator_findings":
        text = f"{c.tool} findings, -{c.penalty_per_finding:g} each"  # type: ignore[attr-defined]
    elif kind == "word_count":
        text = "words: " + ", ".join(f"{t.min_words}+ = {t.points:g}" for t in c.tiers)  # type: ignore[attr-defined]
    elif kind == "code_ratio":
        text = f"code share x {c.scale:g}"  # type: ignore[attr-defined]
    elif kind == "tool_calls":
        text = "tool calls: " + ", ".join(call.name for call in c.calls)  # type: ignore[attr-defined]
    elif kind == "scaled_score":
        text = f"{c.factor:g} x {c.of_rubric} score"  # type: ignore[attr-defined]
    elif kind == "numeric_answer":
        text = f"number {c.expected:g} +/- {c.tolerance:g}"  # type: ignore[attr-defined]
    elif kind == "choice_answer":
        text = f"answer {c.expected}"  # type: ignore[attr-defined]
    elif kind == "external":
        text = "external: " + _short(" ".join(c.cmd))  # type: ignore[attr-defined]
    elif kind == "no_tool_calls":
        text = "no calls to: " + ", ".join(call.name for call in c.calls)  # type: ignore[attr-defined]
    elif kind in ("table_row", "no_table_row"):
        text = f"{label}: " + _short(" + ".join(f"/{p}/" for p in c.match))  # type: ignore[attr-defined]
    elif kind == "table_rows":
        upper = c.max_rows if c.max_rows is not None else "any"  # type: ignore[attr-defined]
        text = f"table rows {c.min_rows}-{upper}"  # type: ignore[attr-defined]
    else:
        text = label
    on = getattr(c, "on", "extracted")
    return text + (f" (on {on})" if on != "extracted" else "")


# --------------------------------------------------------------------------- list


class ListWhat(str, Enum):
    tests = "tests"
    packs = "packs"
    models = "models"
    endpoints = "endpoints"
    runs = "runs"


@app.command("list")
def list_(
    ctx: typer.Context,
    what: ListWhat = typer.Argument(ListWhat.tests, help="tests, packs, models, endpoints or runs"),
    suite: str | None = typer.Option(None, "--suite", help="Only tests in this suite (quick, standard, full, perf)."),
    tag: list[str] = typer.Option([], "--tag", help="Only tests with this tag. Repeatable."),
) -> None:
    """List tests, packs, models, endpoints or run folders."""
    cfg = _load(ctx)
    table = Table(box=box.SIMPLE_HEAD, pad_edge=False)
    if what in (ListWhat.tests, ListWhat.packs):
        catalog = load_catalog(cfg)
        for issue in catalog.issues:
            err_console.print(_issue_text(issue))
        if what == ListWhat.tests:
            tests: list[LoadedTest] = select_tests(catalog, cfg.packs, suite=suite, tags=tag)
            for column in ("ID", "Name", "Category", "Difficulty", "Pack", "Suites"):
                table.add_column(column)
            for lt in tests:
                t = lt.test
                table.add_row(*(Text(v) for v in (t.id, t.name, t.category, t.difficulty, lt.pack.pack.pack, ", ".join(t.suites))))
            _say(f"{len(tests)} test(s)")
        else:
            for column in ("Pack", "Version", "Tests", "Source"):
                table.add_column(column)
            for lp in catalog.packs:
                source = "built in" if lp.builtin else str(lp.path)
                table.add_row(*(Text(v) for v in (lp.pack.pack, lp.pack.version, str(len(lp.pack.tests)), source)))
    elif what == ListWhat.models:
        for column in ("Model", "Endpoint", "Enabled", "Thinking", "Options"):
            table.add_column(column)
        for m in cfg.models:
            options = ", ".join(f"{k}={v}" for k, v in m.options.items())
            cells = (m.name, m.endpoint or cfg.endpoints[0].name, "yes" if m.enabled else "no", m.thinking or "default", options)
            table.add_row(*(Text(v) for v in cells))
        if not cfg.models:
            _say('No models listed. Add models = ["name", ...] to genbench.toml; without it, '
                 "'genbench run' offers the models installed on the server.")
    elif what == ListWhat.runs:
        _print_runs(cfg)
        return
    else:
        for column in ("Endpoint", "Kind", "URL", "Fallbacks"):
            table.add_column(column)
        for ep in cfg.endpoints:
            table.add_row(*(Text(v) for v in (ep.name, ep.kind, ep.url, ", ".join(ep.fallback_urls))))
    console.print(table)


# --------------------------------------------------------------------------- schema / init / new-test


@app.command()
def schema(out: Path | None = typer.Option(None, "--out", "-o", help="Write to this file instead of printing.")) -> None:
    """Print (or write) the JSON Schema for test packs."""
    from genbench.schema import schema_text

    text = schema_text()
    if out is None:
        sys.stdout.write(text)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    _say(f"Wrote {out}")


class Kind(str, Enum):
    ollama = "ollama"
    openai = "openai"


@app.command()
def init(
    directory: Path = typer.Argument(Path("."), help="Folder for genbench.toml, packs/ and schemas/."),
    host: list[str] = typer.Option([], "--host", help="Model server host or URL. Repeatable. Default: $OLLAMA_HOST or localhost."),
    kind: Kind = typer.Option(Kind.ollama, "--kind", help="API the --host servers speak."),
    import_models: Path | None = typer.Option(
        None, "--import-models", help="Take model names from a models.txt (one per line, or pasted 'ollama list' output)."
    ),
    probe_servers: bool = typer.Option(True, "--probe/--no-probe", help="Ask each server which models it has."),
    force: bool = typer.Option(False, "--force", help=f"Overwrite an existing {CONFIG_FILENAME}."),
    full: bool = typer.Option(False, "--full", help="Write every section with its default, not only what differs."),
) -> None:
    """Write a starter genbench.toml for this machine, with packs/ and schemas/ folders.

    You only need one for a remote server or a fixed model list: with no config at all, genbench uses
    Ollama on this machine and offers the models installed there."""
    from genbench.scaffold import init_project

    try:
        result = init_project(
            directory, hosts=host, kind=kind.value, import_models=import_models, probe=probe_servers, force=force,
            full=full,
        )
    except FileExistsError as exc:
        raise _fail(f"{exc} already exists; use --force to overwrite it.") from None
    except (ConfigError, OSError, ValueError) as exc:
        raise _fail(str(exc)) from None
    for note in result.notes:
        _say(note)
    _say(f"Wrote {result.config_path}", "green")
    _say("Next: review it, then run 'genbench validate'.")


@app.command("new-test")
def new_test(
    ctx: typer.Context,
    pack: Path | None = typer.Option(None, "--pack", help="Pack folder (default: packs/my-tests next to genbench.toml)."),
    test_id: str | None = typer.Option(None, "--id", help=TEST_ID_HELP),
    name: str | None = typer.Option(None, "--name", help="Human-readable test name."),
    description: str | None = typer.Option(None, "--description", help="What the model is asked to do."),
    goal: str | None = typer.Option(None, "--goal", help="What a passing answer looks like."),
    category: str | None = typer.Option(None, "--category", help="e.g. coding, reasoning, agentic."),
    expect: list[str] = typer.Option([], "--expect", help="One expectation in plain English. Repeatable."),
    prompt: str | None = typer.Option(None, "--prompt", help="The prompt (you can also edit it in later)."),
) -> None:
    """Start a new test from plain-English sentences; you then attach a check to each expectation."""
    import re

    from genbench.scaffold import SCHEMA_RELPATH, add_test

    cfg = _load(ctx)
    test_id = test_id or typer.prompt("Test id (" + TEST_ID_HELP + ")")
    while not re.fullmatch(r"[a-z0-9][a-z0-9_]*", test_id):
        test_id = typer.prompt("Use lowercase letters, digits and underscores. Test id")
    name = name or typer.prompt("Name")
    description = description or typer.prompt("Description (what the model is asked to do)")
    goal = goal or typer.prompt("Goal (what a passing answer looks like)")
    category = category or typer.prompt("Category", default="coding")
    expectations = [e.strip() for e in expect if e.strip()]
    if not expectations:
        while True:
            entry = typer.prompt(f"Expectation {len(expectations) + 1} (blank to finish)", default="", show_default=False)
            if entry.strip():
                expectations.append(entry.strip())
            elif expectations:
                break
            else:
                _say("Add at least one expectation.")
    pack_dir = pack or cfg.resolve_path("packs") / "my-tests"
    schema_file = cfg.base_dir / SCHEMA_RELPATH
    try:
        path = add_test(
            pack_dir,
            test_id=test_id,
            name=name,
            description=description,
            goal=goal,
            category=category,
            expectations=expectations,
            prompt=prompt,
            schema_file=schema_file if schema_file.is_file() else None,
        )
    except ValidationError as exc:
        raise _fail(format_validation_error(exc, str(pack_dir / "pack.json"))) from None
    except (OSError, ValueError) as exc:
        raise _fail(str(exc)) from None
    _say(f"Added '{test_id}' to {path}", "green")
    _say("Next: write the prompt, replace each TODO check with a real one, then run 'genbench validate'.")
    roots = [cfg.resolve_path(d).resolve() for d in cfg.packs.dirs]
    if not any(root == path.parent.resolve() or root in path.parent.resolve().parents for root in roots):
        _say(f"Note: {path.parent} isn't under [packs].dirs, so genbench won't load it until you add it.", "yellow")


# --------------------------------------------------------------------------- run / regrade / report


class ValidatorMode(str, Enum):
    auto = "auto"
    tool = "tool"
    fallback = "fallback"


def _pick_models(cfg: Config, names: list[str]) -> list:
    from genbench.config import ModelEntry

    if not names:
        return cfg.enabled_models()
    picked = []
    for name in names:
        matches = [m for m in cfg.models if name in (m.name, m.label)]
        picked.extend(matches or [ModelEntry(name=name, endpoint=cfg.endpoints[0].name)])
    return picked


def _interactive() -> bool:
    import sys

    return sys.stdin.isatty()


def _installed_models(cfg: Config, assume_yes: bool) -> list:
    """No models are configured: offer the chat models installed on the first endpoint."""
    from genbench.backends.probe import chat_models, probe
    from genbench.config import ModelEntry

    endpoint = cfg.endpoints[0]
    how = 'list them in genbench.toml as models = ["name", ...] or pass --model NAME'
    answer = probe(endpoint, timeout=5.0)
    if not answer.ok:
        raise _fail(f"No models are listed, and {endpoint.name} could not be asked which it has ({answer.error}). "
                    f"Start the server, or {how}.")
    names = chat_models(answer)
    if not names:
        raise _fail(f"No models are listed, and {endpoint.name} has none installed. Pull one first, e.g. 'ollama pull llama3.2'.")
    _say(f"No models are listed in the config. {endpoint.name} has {len(names)} installed:")
    for name in names:
        _say(f"  {name}")
    left_out = len(answer.models) - len(names)
    if left_out:
        _say(f"  ({left_out} embedding model{'s' if left_out != 1 else ''} left out)")
    if not assume_yes:
        if cfg.run.non_interactive or not _interactive():
            raise _fail(f"Pass --all-models to run all {len(names)}, or --model NAME to pick.")
        if not typer.confirm(f"Run all {len(names)}?", default=False):
            _say(f"Nothing was run. To pick models, {how}.")
            raise typer.Exit(0)
    return [ModelEntry(name=name, endpoint=endpoint.name) for name in names]


def _tests_meta(catalog, test_ids) -> dict:
    meta = {}
    for test_id in test_ids:
        found = catalog.find(test_id)
        if found is not None:
            meta[test_id] = {"name": found.test.name, "description": found.test.description, "goal": found.test.goal}
    return meta


def _duration(seconds: float) -> str:
    """404.84 -> '6m 45s'; 3725 -> '1h 02m 05s'."""
    whole = int(round(seconds))
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s" if hours else f"{minutes}m {secs:02d}s"


def _print_leaderboard(summaries: list, records: list | None = None) -> None:
    from genbench.results import skipped_tests

    not_counted = skipped_tests(records or [])
    if not_counted and summaries:
        _say(f"Not counted (not every model could run them): {', '.join(not_counted)}", "yellow")
    if not summaries:
        _say("No model completed a test, so there is no leaderboard.", "yellow")
        return
    categories = list(dict.fromkeys(c for s in summaries for c in s.categories))
    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, title="Leaderboard", title_justify="left")
    titles = [c.replace("_", " ").replace("-", " ").title() for c in categories]
    for column in ("Rank", "Model", "Overall", "Passed", "Accuracy", *titles, "TTFT (s)", "Tok/s"):
        table.add_column(column, justify="left" if column == "Model" else "right")
    for rank, s in enumerate(summaries, start=1):
        passed = f"{s.passed}/{s.with_mark}" if s.with_mark else "-"
        cells = [f"#{rank}", s.model, f"{s.overall}", passed, f"{s.accuracy}",
                 *[f"{s.categories[c]}" if c in s.categories else "-" for c in categories],
                 f"{s.ttft_s}", f"{s.tokens_per_sec}"]
        table.add_row(*(Text(c) for c in cells))
    console.print(table)


@app.command()
def run(
    ctx: typer.Context,
    model: list[str] = typer.Option([], "--model", "-m", help="Only this model (from genbench.toml, or any model on the first endpoint). Repeatable."),
    test: list[str] = typer.Option([], "--test", "-t", help="Only this test id or alias. Repeatable."),
    suite: str | None = typer.Option(None, "--suite", help="quick, standard, full or perf."),
    tag: list[str] = typer.Option([], "--tag", help="Only tests with this tag. Repeatable."),
    repeats: int | None = typer.Option(None, "--repeats", min=1, help="Override run.repeats."),
    all_models: bool = typer.Option(
        False, "--all-models", help="When the config lists no models: run every model installed on the server, without asking."
    ),
) -> None:
    """Run the selected tests on every enabled model; results and reports go to a new run folder.

    With no models in the config, genbench offers the ones installed on the server."""
    from math import fsum

    from genbench.backends.base import MODEL_TYPE_LABELS
    from genbench.report import write_reports
    from genbench.results import RunStore, new_run_dir
    from genbench.runner import RunEvents, RunPlan, execute

    cfg = _load(ctx)
    catalog = load_catalog(cfg)
    errors = [i for i in validate_catalog(catalog, cfg) if i.level == "error"]
    if errors:
        for issue in errors:
            console.print(_issue_text(issue))
        raise _fail("Fix the errors above before running (genbench validate shows them too).")
    tests = select_tests(catalog, cfg.packs, suite=suite, tags=tag, ids=test)
    if not tests:
        raise _fail("No tests match those filters. 'genbench list tests' shows what's available.")
    models = _pick_models(cfg, model)
    if not models and cfg.models:
        raise _fail("Every model in the config has enabled = false. Enable one, or pass --model NAME.")
    if not models:
        models = _installed_models(cfg, all_models)
    for m in models:
        endpoint = cfg.endpoint(m.endpoint)
        if endpoint.kind != "ollama":
            raise _fail(f"Endpoint '{endpoint.name}' is '{endpoint.kind}'; OpenAI-compatible servers arrive in Phase 2.")

    run_dir = new_run_dir(cfg.resolve_path(cfg.run.output_dir), cfg.run.name)
    plan = RunPlan(tests, models, repeats or cfg.run.repeats, run_dir)
    rule = "=" * 80
    _say(rule)
    _say(f"genbench run '{cfg.run.name}': {len(models)} model(s) x {len(tests)} test(s), {plan.repeats} repeat(s)")
    _say(f"Output: {run_dir}")
    _say(rule)

    line_open = {"value": False}

    def model_start(i, n, m, endpoint, kind=None):
        _say("")
        _say(f"[{i}/{n}] {m.label} on {endpoint}" + (f"  ({MODEL_TYPE_LABELS[kind]})" if kind else ""), "bold")
        console.print(Text("  Warming up..."), end="")
        line_open["value"] = True

    def warmup_done(m, info):
        if info.error:
            console.print(Text(f" failed: {info.error}", style="yellow"))
        else:
            console.print(Text(f" ready (cold load {info.cold_load_s}s | VRAM {info.label})"))
            if info.gpu_pct is not None and info.gpu_pct < 100:
                _say(f"  Warning: only {info.gpu_pct:g}% of the model fits in VRAM; the rest runs on the CPU, so "
                     "speeds will be low. Something else may be using the GPU.", "yellow")
            if info.others:
                _say(f"  Warning: other models are loaded on the server too: {', '.join(info.others)}", "yellow")
        line_open["value"] = False

    def model_error(m, message):
        if line_open["value"]:
            console.print()
        _say(f"  Skipping {m.label}: {message}", "red")
        line_open["value"] = False

    def model_done(m, records):
        ran = [r for r in records if r.ok]
        if not ran:
            return
        total = fsum(r.total_s or 0.0 for r in ran)
        speed = fsum(r.tokens_per_sec or 0.0 for r in ran) / len(ran)
        parts = [f"Total time {total:.2f}s" + (f" ({_duration(total)})" if total >= 60 else ""),
                 f"average {speed:.2f} tok/s", f"{len(ran)} test{'s' if len(ran) != 1 else ''} run"]
        skipped = sum(1 for r in records if r.skipped)
        errors = sum(1 for r in records if r.error)
        parts[-1] += (f", {skipped} skipped" if skipped else "") + (f", {errors} failed to run" if errors else "")
        _say("  " + " | ".join(parts), "bold")

    def test_start(lt, repeat, count):
        label = lt.test.name + (f" [{repeat}/{count}]" if count > 1 else "")
        console.print(Text(f"  {label[:58]:<58} "), end="")
        line_open["value"] = True

    def test_done(record):
        if not line_open["value"]:
            return
        line_open["value"] = False
        if record.skipped:
            console.print(Text(f"SKIP  {record.skipped}", style="yellow"))
            return
        if record.error:
            console.print(Text(f"ERROR {record.error[:100]}", style="red"))
            return
        verdict = "PASS" if record.passed else ("FAIL" if record.passed is False else "done")
        style = "green" if record.passed else ("red" if record.passed is False else "")
        text = Text(f"{verdict:<5}", style=style)
        text.append(f" {record.total_s}s | TTFT {record.warm_ttft_s}s | {record.tokens_per_sec} tok/s | "
                    f"acc {record.accuracy} | quality {record.quality}")
        console.print(text)
        for warning in record.warnings:
            _say("  Warning: " + warning, "yellow")

    events = RunEvents(model_start=model_start, warmup_done=warmup_done, test_start=test_start,
                       test_done=test_done, model_error=model_error, model_done=model_done)
    interrupted = False
    try:
        execute(cfg, plan, events)
    except KeyboardInterrupt:
        interrupted = True
        console.print()
        _say("Interrupted; writing reports for the attempts that finished.", "yellow")
    _say("")
    summaries, written = write_reports(run_dir, plan.records, cfg, RunStore(run_dir).read_meta())
    _print_leaderboard(summaries, plan.records)
    for path in [run_dir / "results.jsonl", *written]:
        _say(f"Wrote {path}")
    if interrupted:
        raise typer.Exit(130)


@app.command()
def regrade(
    ctx: typer.Context,
    source: Path | None = typer.Argument(None, help="A run folder to re-grade with today's packs and config."),
    from_python_csv: Path | None = typer.Option(
        None, "--from-python-csv", help="A results CSV written by ollama_local_benchmark.py."
    ),
    validators: ValidatorMode = typer.Option(
        ValidatorMode.auto, "--validators", help="Old CSVs: auto follows the saved notes; tool or fallback forces one."
    ),
    tolerance: float = typer.Option(0.1, "--tolerance", help="Old CSVs: largest allowed difference from the old scores."),
) -> None:
    """Re-score saved answers without calling a model: a run folder, or an old CSV for the parity check."""
    from genbench.grade.engine import GradeContext, grade_test
    from genbench.grade.validators import ValidatorRunner
    from genbench.legacy import regrade_legacy
    from genbench.report import write_reports
    from genbench.results import RunStore, artifact_dir, new_run_dir, slug
    from genbench.runner import add_warnings

    if (source is None) == (from_python_csv is None):
        raise _fail("Give either a run folder or --from-python-csv FILE.")
    cfg = _load(ctx)
    catalog = load_catalog(cfg)
    root = cfg.resolve_path(cfg.run.output_dir)

    if from_python_csv is not None:
        run_dir = new_run_dir(root, f"regrade-{slug(from_python_csv.stem)}")
        try:
            records, parity = regrade_legacy(from_python_csv, catalog, cfg, run_dir, validators.value)
        except (OSError, ValueError) as exc:
            raise _fail(str(exc)) from None
        store = RunStore(run_dir)
        store.write_all(records)
        meta = {"run_id": run_dir.name, "source": str(from_python_csv), "kind": "regrade of an old CSV", "status": "completed",
                "tests": _tests_meta(catalog, dict.fromkeys(r.test_id for r in records))}
        store.write_meta(meta)
        table = Table(box=box.SIMPLE_HEAD, pad_edge=False, title="Parity with the original scores", title_justify="left")
        for column in ("Model", "Test", "Old acc", "New acc", "Old cat", "New cat", "Gap"):
            table.add_column(column, justify="left" if column in ("Model", "Test") else "right")
        for p in parity:
            gap = p.worst_gap
            table.add_row(Text(p.model), Text(p.test_id), Text(f"{p.old_accuracy}"), Text(f"{p.new_accuracy}"),
                          Text(f"{p.old_category}"), Text(f"{p.new_category}"),
                          Text(f"{gap:g}", style="green" if gap <= tolerance + 1e-9 else "red"))
        console.print(table)
        matched = sum(1 for p in parity if p.worst_gap <= tolerance + 1e-9)
        summaries, written = write_reports(run_dir, records, cfg, meta)
        _print_leaderboard(summaries, records)
        for path in [run_dir / "results.jsonl", *written]:
            _say(f"Wrote {path}")
        style = "green" if matched == len(parity) else "red"
        _say(f"{matched}/{len(parity)} rows within {tolerance:g} of the original scores.", style)
        if matched != len(parity):
            raise typer.Exit(1)
        return

    assert source is not None
    try:
        old = RunStore(source).load()
    except (OSError, ValueError) as exc:
        raise _fail(str(exc)) from None
    run_dir = new_run_dir(root, f"regrade-{source.name}")
    runner = ValidatorRunner(cfg.validators)
    warmups = RunStore(source).read_meta().get("warmups") or {}
    records = []
    for record in old:
        record = record.model_copy(deep=True)
        record.run_id = run_dir.name
        found = catalog.find(record.test_id)
        if record.ok and found is not None:
            ctx_ = GradeContext(runner, artifact_dir(run_dir, record.model, record.test_id, record.repeat, 2 if record.repeat > 1 else 1),
                                found.pack.dir, cfg.scoring.pass_mark, record.model)
            kept = [part for part in record.notes.split(" | ") if "server may have cut it" in part]
            record.apply_grade(grade_test(found.test, record.response, ctx_, thinking=record.thinking,
                                          native_calls=record.tool_calls), run_dir)
            if record.context_tokens is None:  # runs saved before the window was stored per attempt
                record.context_tokens = (warmups.get(record.model) or {}).get("context_tokens")
            add_warnings(record)
            if not record.warnings and kept:
                record.warnings = kept
                record.notes = " | ".join([record.notes, *kept])
        elif found is None:
            _say(f"Test '{record.test_id}' no longer exists; keeping its old grade.", "yellow")
        records.append(record)
    store = RunStore(run_dir)
    store.write_all(records)
    meta = {**RunStore(source).read_meta(), "run_id": run_dir.name, "regraded_from": str(source),
            "kind": "regrade", "status": "completed"}
    meta["tests"] = {**meta.get("tests", {}), **_tests_meta(catalog, dict.fromkeys(r.test_id for r in records))}
    store.write_meta(meta)
    summaries, written = write_reports(run_dir, records, cfg, meta)
    _print_leaderboard(summaries, records)
    for path in [run_dir / "results.jsonl", *written]:
        _say(f"Wrote {path}")


@app.command()
def report(ctx: typer.Context, run_dir: Path = typer.Argument(..., help="A run folder, e.g. runs/20261109-201500_blueboy-p6000.")) -> None:
    """Rebuild a run folder's reports from its results.jsonl (after changing weights or the title, say)."""
    from genbench.report import write_reports
    from genbench.results import RunStore

    cfg = _load(ctx)
    store = RunStore(run_dir)
    try:
        records = store.load()
    except (OSError, ValueError) as exc:
        raise _fail(str(exc)) from None
    summaries, written = write_reports(run_dir, records, cfg, store.read_meta())
    _print_leaderboard(summaries, records)
    for path in written:
        _say(f"Wrote {path}")


# --------------------------------------------------------------------------- run folders: list and clean


def _runs_table(infos: list, title: str | None = None) -> Table:
    from genbench.results import human_size

    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, title=title, title_justify="left")
    for column in ("Run folder", "Status", "Started", "Models", "Attempts", "Size"):
        table.add_column(column, justify="right" if column in ("Models", "Attempts", "Size") else "left")
    styles = {"completed": "green", "running": "cyan"}
    for info in infos:
        attempts = f"{info.attempts}/{info.expected}" if info.expected else str(info.attempts)
        table.add_row(
            Text(info.path.name), Text(info.status, style=styles.get(info.status, "yellow")),
            Text(info.started.strftime("%Y-%m-%d %H:%M")), Text(str(info.models or "-")), Text(attempts),
            Text(human_size(info.size)),
        )
    return table


def _print_runs(cfg: Config) -> None:
    from genbench.results import scan_runs

    root = cfg.resolve_path(cfg.run.output_dir)
    infos = scan_runs(root)
    if not infos:
        _say(f"No run folders in {root}")
        return
    console.print(_runs_table(infos))
    _say(f"{len(infos)} run(s) in {root}")


@app.command()
def clean(
    ctx: typer.Context,
    runs: list[Path] | None = typer.Argument(None, help="Run folders to remove. Default: every incomplete run."),
    incomplete: bool = typer.Option(False, "--incomplete", help="Remove interrupted, failed, empty and stopped runs (the default)."),
    all_runs: bool = typer.Option(False, "--all", help="Remove every run folder except one still running."),
    older_than: str | None = typer.Option(None, "--older-than", metavar="AGE", help="Remove runs started before AGE ago, e.g. 30d, 12h."),
    keep: int | None = typer.Option(None, "--keep", min=0, help="Keep the N newest completed runs; remove older completed ones."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only show what would be removed."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Don't ask before deleting."),
) -> None:
    """Delete run folders under [run].output_dir. With no options: only the incomplete ones."""
    from datetime import datetime

    from genbench.results import (
        INCOMPLETE,
        human_size,
        is_run_dir,
        parse_age,
        remove_run,
        run_info,
        scan_runs,
    )

    cfg = _load(ctx)
    root = cfg.resolve_path(cfg.run.output_dir)
    now = datetime.now()

    if runs:
        chosen = []
        for given in runs:
            path = given if given.exists() or given.is_absolute() else root / given
            if not is_run_dir(path):
                raise _fail(f"{given} isn't a genbench run folder (no run.json or results.jsonl), so nothing was deleted.")
            if path.resolve() == root.resolve():
                raise _fail("That is the runs folder itself; name the run folders inside it.")
            chosen.append(run_info(path, now))
    else:
        try:
            age = parse_age(older_than) if older_than else None
        except ValueError as exc:
            raise _fail(str(exc)) from None
        infos = scan_runs(root, now)
        default = not (all_runs or age or keep is not None)
        picked: dict[Path, object] = {}
        for info in infos:
            if info.status == "running":
                continue  # possibly still going in another window
            if all_runs or ((incomplete or default) and info.status in INCOMPLETE) or (age and now - info.started > age):
                picked[info.path] = info
        if keep is not None:
            completed = [i for i in infos if i.status == "completed"]
            for info in completed[keep:]:
                picked[info.path] = info
        chosen = list(picked.values())
        active = [i for i in infos if i.status == "running"]
        if active:
            _say(f"Leaving {len(active)} run(s) marked as running; name a folder to remove it anyway.", "dim")

    if not chosen:
        _say("Nothing to clean." + ("" if runs else f" ({root})"))
        return
    total = sum(i.size for i in chosen)
    console.print(_runs_table(chosen, "To delete"))
    if dry_run:
        _say(f"Dry run: would delete {len(chosen)} run folder(s), {human_size(total)}.")
        return
    if not yes and not typer.confirm(f"Delete {len(chosen)} run folder(s), {human_size(total)}?", default=False):
        _say("Nothing deleted.")
        return
    failed: list[str] = []
    freed = 0
    for info in chosen:
        try:
            remove_run(info.path)
            freed += info.size
        except (OSError, ValueError) as exc:
            failed.append(f"{info.path.name}: {exc}")
    _say(f"Deleted {len(chosen) - len(failed)} run folder(s), freed {human_size(freed)}.", "yellow" if failed else "green")
    for line in failed:
        _say(f"  Couldn't delete {line}", "red")
    if failed:
        raise typer.Exit(1)
