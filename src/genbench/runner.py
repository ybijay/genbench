"""Run tests against models: warm up each model, stream every test, grade it, save every attempt."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from genbench import __version__
from genbench.backends import BackendError, Warmup, make_backend
from genbench.config import Config, ModelEntry
from genbench.context import estimate_tokens
from genbench.context import generate as generate_context
from genbench.grade.engine import GradeContext, grade_test
from genbench.grade.validators import ValidatorRunner
from genbench.pack.loader import LoadedTest
from genbench.results import AttemptRecord, RunStore, artifact_dir, test_hash

VAR = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


@dataclass
class RunEvents:
    """Callbacks the CLI uses for live progress; all optional."""

    model_start: Callable[[int, int, ModelEntry, str, str | None], None] = lambda i, n, model, endpoint, kind: None
    model_done: Callable[[ModelEntry, list[AttemptRecord]], None] = lambda model, records: None
    warmup_done: Callable[[ModelEntry, Warmup], None] = lambda model, info: None
    test_start: Callable[[LoadedTest, int, int], None] = lambda lt, repeat, repeats: None
    test_done: Callable[[AttemptRecord], None] = lambda record: None
    model_error: Callable[[ModelEntry, str], None] = lambda model, message: None


@dataclass
class RunPlan:
    tests: list[LoadedTest]
    models: list[ModelEntry]
    repeats: int
    run_dir: Path
    records: list[AttemptRecord] = field(default_factory=list)


def render_prompt(text: str, variables: dict[str, Any]) -> str:
    return VAR.sub(lambda m: str(variables[m.group(1)]) if m.group(1) in variables else m.group(0), text)


TRUNCATION_SHARE = 0.95  # a prompt this close to the context window may have been cut by the server
CONTEXT_MARGIN = 1.05  # the size estimate runs up to 5% low (Gemma)
ANSWER_RESERVE = 2000  # tokens kept free for thinking and the answer
WINDOW_SLACK = 4  # a reply that ends this close to the window's last token was stopped by it
WINDOW_OVERRUN = 64  # more than this past the window means the server slid the window instead of stopping


def prompt_variables(lt: LoadedTest) -> dict[str, Any]:
    variables = dict(lt.test.vars)
    if lt.test.context is not None:
        variables["context"] = generate_context(lt.test.context)
    return variables


def build_messages(lt: LoadedTest) -> list[dict[str, Any]]:
    test = lt.test
    variables = prompt_variables(lt)
    if test.messages is not None:
        messages = [m.model_dump(exclude_none=True) for m in test.messages]
        for m in messages:
            m["content"] = render_prompt(m.get("content", ""), variables)
    else:
        messages = [{"role": "user", "content": render_prompt(lt.prompt_text() or "", variables)}]
    if test.system:
        messages.insert(0, {"role": "system", "content": render_prompt(test.system, variables)})
    return messages


def context_needed(messages: list[dict[str, Any]]) -> int:
    """The context window a prompt needs, in tokens: its estimated size plus room to answer."""
    size = sum(estimate_tokens(str(m.get("content") or "")) for m in messages)
    return int(size * CONTEXT_MARGIN) + ANSWER_RESERVE


def context_shortfall(messages: list[dict[str, Any]], context_tokens: int | None) -> str | None:
    """Why the test can't run on a model with this context window, or None when it fits (or the window is unknown)."""
    if not context_tokens:
        return None
    needed = context_needed(messages)
    if needed > context_tokens:
        return f"Needs a context window of about {needed:,} tokens; this model has {context_tokens:,}"
    return None


def truncation_warning(prompt_tokens: int, context_tokens: int | None) -> str | None:
    if context_tokens and prompt_tokens >= TRUNCATION_SHARE * context_tokens:
        return (f"Prompt used {prompt_tokens:,} of {context_tokens:,} context tokens, so the server may have cut it "
                "and scores may be unfair")
    return None


def cut_off_warning(prompt_tokens: int, completion_tokens: int, context_tokens: int | None, done_reason: str,
                    answered: bool) -> str | None:
    """The reply stopped because it ran out of room, not because the model was done."""
    effect = "so the answer may be cut short" if answered else "so it never answered"
    if context_tokens and prompt_tokens + completion_tokens > context_tokens + WINDOW_OVERRUN:
        # Some servers don't stop at a full window: they drop the oldest text and let the model carry on.
        return (f"Went past the context window: {prompt_tokens:,} prompt + {completion_tokens:,} generated tokens is more "
                f"than the {context_tokens:,}-token window, so the server dropped the oldest text and the model lost the "
                "start of its prompt")
    if context_tokens and prompt_tokens + completion_tokens >= context_tokens - WINDOW_SLACK:
        return (f"Ran out of context window: {prompt_tokens:,} prompt + {completion_tokens:,} generated tokens "
                f"filled the {context_tokens:,}-token window, {effect}")
    if done_reason == "length":
        return f"Stopped at the output limit after {completion_tokens:,} generated tokens, {effect}"
    return None


def no_final_message_warning(record: AttemptRecord) -> str | None:
    """The stream stopped without the server's closing message, which carries the token counts."""
    wrote = record.thinking or record.response
    if not wrote or record.done_reason or record.prompt_tokens or record.completion_tokens:
        return None
    effect = "the answer may be cut short" if record.response.strip() else "it never answered"
    return (f"The server stopped mid-reply without its closing message, after about {len(record.thinking):,} characters "
            f"of thinking, so {effect}. It most likely ran out of context window or the model stopped on the server")


def window_warnings(record: AttemptRecord) -> list[str]:
    """What the context window did to this attempt, worked out from what the record holds."""
    found = [
        truncation_warning(record.prompt_tokens, record.context_tokens),
        cut_off_warning(record.prompt_tokens, record.completion_tokens, record.context_tokens, record.done_reason,
                        bool(record.response.strip())),
        no_final_message_warning(record),
    ]
    return [warning for warning in found if warning]


def add_warnings(record: AttemptRecord) -> None:
    record.warnings = window_warnings(record)
    record.notes = " | ".join([record.notes, *record.warnings])


def run_meta(cfg: Config, plan: RunPlan, run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "genbench_version": __version__,
        "config": str(cfg.source) if cfg.source else None,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "models": [m.model_dump() for m in plan.models],
        "endpoints": [e.model_dump() for e in cfg.endpoints],
        "repeats": plan.repeats,
        "status": "running",
        "expected_attempts": len(plan.models) * sum(lt.test.repeats or plan.repeats for lt in plan.tests),
        "tests": {
            lt.test.id: {
                "name": lt.test.name,
                "description": lt.test.description,
                "goal": lt.test.goal,
                "pack": lt.pack.pack.pack,
                "pack_version": lt.pack.pack.version,
                "hash": test_hash(lt),
            }
            for lt in plan.tests
        },
    }


def execute(cfg: Config, plan: RunPlan, events: RunEvents | None = None) -> list[AttemptRecord]:
    events = events or RunEvents()
    store = RunStore(plan.run_dir)
    run_id = plan.run_dir.name
    meta = run_meta(cfg, plan, run_id)
    store.write_meta(meta)
    validators = ValidatorRunner(cfg.validators)
    backends: dict[str, Any] = {}
    hashes = {lt.test.id: meta["tests"][lt.test.id]["hash"] for lt in plan.tests}
    status = "failed"
    try:
        for index, model in enumerate(plan.models, start=1):
            endpoint = cfg.endpoint(model.endpoint)
            backend, failure = None, None
            try:
                backend = backends.get(endpoint.name) or make_backend(endpoint)
                backends[endpoint.name] = backend
                if backend.base_url is None:
                    backend.connect()
            except BackendError as exc:
                failure = str(exc)
            # Dense or MoE: what genbench.toml says, else what the server reports about the model.
            kind = model.type or (backend.model_type(model.name) if backend is not None and failure is None else None)
            events.model_start(index, len(plan.models), model, endpoint.name, kind)
            if backend is None or failure is not None:
                events.model_error(model, failure or "no backend")
                for lt in plan.tests:
                    record = _record(run_id, model, endpoint.name, lt, hashes, 1)
                    record.error = failure
                    _save(store, plan, record, events)
                continue
            options = {**cfg.defaults.options, **model.options}
            warm = backend.warmup(model.name, options, model.thinking)
            if warm.error and model.thinking and not backend.warmup(model.name, options).error:
                # The model loads fine without the setting, so the server refused the setting itself.
                failure = f"The server refused thinking = \"{model.thinking}\" for this model: {warm.error}"
                events.model_error(model, failure)
                for lt in plan.tests:
                    record = _record(run_id, model, endpoint.name, lt, hashes, 1)
                    record.error = failure
                    _save(store, plan, record, events)
                continue
            warm.model_type = kind
            first = len(plan.records)
            meta.setdefault("warmups", {})[model.label] = asdict(warm)
            store.write_meta(meta)
            events.warmup_done(model, warm)
            for lt in plan.tests:
                repeats = lt.test.repeats or plan.repeats
                messages = build_messages(lt)
                window = lt.test.options.get("num_ctx") or warm.context_tokens
                too_big = context_shortfall(messages, window)
                for repeat in range(1, repeats + 1):
                    events.test_start(lt, repeat, repeats)
                    record = _record(run_id, model, endpoint.name, lt, hashes, repeat)
                    record.cold_load_s, record.vram_label = warm.cold_load_s, warm.label
                    if too_big:
                        record.skipped = too_big
                        _save(store, plan, record, events)
                        continue
                    try:
                        attempt = backend.chat(
                            model.name,
                            messages,
                            {**options, **lt.test.options},
                            tools=lt.test.tools,
                            fmt=lt.test.format,
                            timeout_s=lt.test.timeout_s,
                            thinking=model.thinking,
                        )
                    except BackendError as exc:
                        record.error = str(exc)
                        _save(store, plan, record, events)
                        continue
                    for name in ("total_s", "wall_ttft_s", "warm_ttft_s", "first_content_s", "first_thinking_s",
                                 "tokens_per_sec", "prompt_eval_tps", "prompt_tokens", "completion_tokens", "load_s",
                                 "done_reason"):
                        setattr(record, name, getattr(attempt, name))
                    record.context_tokens = window
                    record.response, record.thinking, record.tool_calls = attempt.content, attempt.thinking, attempt.tool_calls
                    ctx = GradeContext(
                        validators=validators,
                        workdir=artifact_dir(plan.run_dir, model.label, lt.test.id, repeat, repeats),
                        pack_dir=lt.pack.dir,
                        default_pass_mark=cfg.scoring.pass_mark,
                        model=model.label,
                    )
                    grade = grade_test(lt.test, attempt.content, ctx, thinking=attempt.thinking, native_calls=attempt.tool_calls)
                    record.apply_grade(grade, plan.run_dir)
                    add_warnings(record)
                    _save(store, plan, record, events)
            events.model_done(model, plan.records[first:])
            if cfg.run.unload_between_models and index < len(plan.models):
                following = plan.models[index]
                same_again = following.name == model.name and cfg.endpoint(following.endpoint).name == endpoint.name
                if not same_again:  # the same model with another thinking setting: keep it loaded
                    backend.unload(model.name)
        status = "completed"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        meta["status"] = status
        meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
        meta["attempts"] = len(plan.records)
        store.write_meta(meta)
        for backend in backends.values():
            backend.close()
    return plan.records


def _record(run_id: str, model: ModelEntry, endpoint: str, lt: LoadedTest, hashes: dict[str, str], repeat: int) -> AttemptRecord:
    test = lt.test
    return AttemptRecord(
        run_id=run_id,
        model=model.label,
        thinking_setting=model.thinking,
        endpoint=endpoint,
        test_id=test.id,
        test_name=test.name,
        pack=lt.pack.pack.pack,
        pack_version=lt.pack.pack.version,
        test_hash=hashes[test.id],
        category=test.category,
        difficulty=test.difficulty,
        repeat=repeat,
        started_at=datetime.now().isoformat(timespec="seconds"),
    )


def _save(store: RunStore, plan: RunPlan, record: AttemptRecord, events: RunEvents) -> None:
    store.append(record)
    plan.records.append(record)
    events.test_done(record)
