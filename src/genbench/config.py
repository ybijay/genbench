"""Run configuration: genbench.toml layered with GENBENCH_* environment variables and --set flags.

Precedence, lowest to highest: built-in defaults, the TOML file, environment variables, --set flags.
Nothing about a particular machine is built in: with no config file, genbench talks to the Ollama
server named by OLLAMA_HOST, or to http://localhost:11434.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python 3.10
    import tomli as tomllib

CONFIG_FILENAME = "genbench.toml"
ENV_CONFIG = "GENBENCH_CONFIG"
ENV_PREFIX = "GENBENCH_"
OLLAMA_PORT = 11434


class ConfigError(Exception):
    """A config that can't be read or doesn't validate. The message is ready to show to the user."""


def normalize_url(url: str, default_port: int | None = None) -> str:
    """'blueboy' -> 'http://blueboy:11434' when default_port is 11434; trailing slashes dropped."""
    url = url.strip()
    if not re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", url):
        url = "http://" + url
    parts = urlsplit(url)
    if default_port is not None and parts.hostname and parts.port is None:
        parts = parts._replace(netloc=f"{parts.netloc}:{default_port}")
    return parts.geturl().rstrip("/")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PassMark(_Strict):
    """Scores a test must reach to count as passed. Every mark that is set must hold."""

    accuracy: float | None = Field(None, ge=0, le=100)
    quality: float | None = Field(None, ge=0, le=100)

    @model_validator(mode="after")
    def _at_least_one(self) -> PassMark:
        if self.accuracy is None and self.quality is None:
            raise ValueError("set 'accuracy', 'quality' or both")
        return self

    def describe(self) -> str:
        marks = [("accuracy", self.accuracy), ("quality", self.quality)]
        return " and ".join(f"{name} >= {value:g}" for name, value in marks if value is not None)


class RunSettings(_Strict):
    name: str = "local"
    output_dir: str = "runs"
    repeats: int = Field(1, ge=1)
    seed: int = 42
    non_interactive: bool = False
    unload_between_models: bool = True
    failed_test_score: float = Field(0.0, ge=0, le=100)


class Endpoint(_Strict):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    kind: Literal["ollama", "openai"] = "ollama"
    url: str = Field(min_length=1)
    fallback_urls: list[str] = Field(default_factory=list)
    timeout_s: float = Field(900.0, gt=0)
    api_key_env: str | None = None
    telemetry_cmd: list[str] | None = None
    reasoning_field: str | None = None
    stream_usage: bool = True

    @model_validator(mode="after")
    def _normalize(self) -> Endpoint:
        port = OLLAMA_PORT if self.kind == "ollama" else None
        self.url = normalize_url(self.url, port)
        self.fallback_urls = [normalize_url(u, port) for u in self.fallback_urls]
        return self

    def urls(self) -> list[str]:
        """The main URL, then fallbacks, without repeats."""
        return list(dict.fromkeys([self.url, *self.fallback_urls]))


def default_endpoint(env: Mapping[str, str] | None = None) -> Endpoint:
    """$OLLAMA_HOST if set (a server bind address like 0.0.0.0 means this machine), else localhost."""
    env = os.environ if env is None else env
    host = env.get("OLLAMA_HOST", "").strip()
    if host:
        parts = urlsplit(host if "://" in host else "http://" + host)
        if parts.hostname in ("0.0.0.0", "::"):
            host = "localhost" + (f":{parts.port}" if parts.port else "")
    return Endpoint(name="local", kind="ollama", url=host or "localhost")


class ModelEntry(_Strict):
    name: str = Field(min_length=1)
    endpoint: str | None = Field(None, description="Endpoint name; default: the first endpoint")
    options: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    type: Literal["dense", "moe"] | None = Field(
        None, description="Dense or MoE (mixture of experts). Leave out and genbench reads it from the server"
    )
    thinking: Literal["off", "on", "low", "medium", "high"] | None = Field(
        None,
        description="Ask the server to switch the model's thinking off or on, or to a level. Leave out (or "
        "'default') for the model's own default. List a model once per setting to compare them",
    )

    @field_validator("thinking", mode="before")
    @classmethod
    def _thinking_words(cls, value: Any) -> Any:
        if isinstance(value, bool):
            return "on" if value else "off"
        if isinstance(value, str):
            value = value.strip().lower()
            return None if value in ("default", "") else value
        return value

    @property
    def label(self) -> str:
        """The name shown in results: the model, plus its thinking setting when one is set."""
        return f"{self.name} (thinking {self.thinking})" if self.thinking else self.name


class Defaults(_Strict):
    options: dict[str, Any] = Field(default_factory=lambda: {"temperature": 0.0, "keep_alive": "15m"})


class PackSettings(_Strict):
    builtin: bool = Field(True, description="Load the tests that ship with genbench")
    dirs: list[str] = Field(
        default_factory=lambda: ["packs"],
        description="Folders with your own packs. The default, packs/ next to the config, is optional",
    )
    suite: Literal["quick", "standard", "full", "perf"] | None = None
    include_tags: list[str] = Field(default_factory=list)
    exclude_tags: list[str] = Field(default_factory=list)
    include_ids: list[str] = Field(default_factory=list)


class ValidatorTool(_Strict):
    builtin: Literal["ast"] | None = None
    cmd: list[str] | None = Field(None, min_length=1)
    timeout_s: float = Field(10.0, gt=0)

    @model_validator(mode="after")
    def _one_kind(self) -> ValidatorTool:
        if (self.builtin is None) == (self.cmd is None):
            raise ValueError("set exactly one of 'builtin' or 'cmd'")
        return self


GIT_BASH_PLACES = (
    "C:/Program Files/Git/bin/bash.exe",
    "C:/Program Files (x86)/Git/bin/bash.exe",
    "~/AppData/Local/Programs/Git/bin/bash.exe",
)


def default_bash_cmd(platform: str | None = None) -> list[str]:
    """Git Bash on Windows when it's installed (WSL's bash can't read Windows paths), else bash."""
    if (platform or sys.platform) == "win32":
        for place in GIT_BASH_PLACES:
            path = Path(place).expanduser()
            if path.is_file():
                return [path.as_posix(), "-n", "{file}"]
    return ["bash", "-n", "{file}"]


def _default_tools() -> dict[str, dict[str, Any]]:
    return {"python": {"builtin": "ast"}, "bash": {"cmd": default_bash_cmd()}}


class ValidatorSettings(_Strict):
    on_missing: Literal["fallback", "skip", "zero"] = "fallback"
    tools: dict[str, ValidatorTool] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _flatten(cls, data: Any) -> Any:
        """Tools may sit directly under [validators] (python = {...}) or under [validators.tools]."""
        if not isinstance(data, Mapping):
            return data
        data = dict(data)
        tools = {**_default_tools(), **dict(data.pop("tools", None) or {})}
        on_missing = data.pop("on_missing", None)
        tools.update(data)
        out: dict[str, Any] = {"tools": tools}
        if on_missing is not None:
            out["on_missing"] = on_missing
        return out


WEIGHT_KEY = re.compile(r"^(accuracy|pass_rate|speed\.tokens_per_sec|speed\.ttft|category\.[a-z0-9][a-z0-9_-]*)$")


class Scoring(_Strict):
    ttft_metric: Literal["warm", "wall", "first_content"] = "warm"
    normalization: Literal["relative", "absolute"] = "relative"
    reference: dict[str, float] = Field(default_factory=lambda: {"tokens_per_sec": 50.0, "ttft_s": 0.5})
    pass_mark: PassMark = Field(default_factory=lambda: PassMark(accuracy=70))
    difficulty_weights: dict[Literal["easy", "medium", "hard", "expert"], float] = Field(
        default_factory=lambda: {"easy": 0.5, "medium": 1.0, "hard": 1.5, "expert": 2.0}
    )
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "accuracy": 0.30,
            "category.coding": 0.25,
            "category.architecture": 0.20,
            "speed.tokens_per_sec": 0.15,
            "speed.ttft": 0.10,
        }
    )

    @field_validator("weights")
    @classmethod
    def _check_weights(cls, value: dict[str, float]) -> dict[str, float]:
        bad = [key for key in value if not WEIGHT_KEY.match(key)]
        if bad:
            raise ValueError(
                f"unknown weight key(s) {', '.join(bad)}; use accuracy, pass_rate, "
                "speed.tokens_per_sec, speed.ttft or category.<name>"
            )
        if any(v < 0 for v in value.values()):
            raise ValueError("weights can't be negative")
        if value and sum(value.values()) <= 0:
            raise ValueError("weights must add up to more than 0")
        return value


class Report(_Strict):
    formats: list[Literal["markdown", "csv", "jsonl", "json", "html"]] = Field(
        default_factory=lambda: ["markdown", "csv", "jsonl"]
    )
    title: str = "Local Benchmark Leaderboard"
    template: str | None = None


class Config(_Strict):
    run: RunSettings = Field(default_factory=RunSettings)
    endpoints: list[Endpoint] = Field(default_factory=list)
    models: list[ModelEntry] = Field(default_factory=list)
    defaults: Defaults = Field(default_factory=Defaults)
    packs: PackSettings = Field(default_factory=PackSettings)
    validators: ValidatorSettings = Field(default_factory=lambda: ValidatorSettings.model_validate({}))
    scoring: Scoring = Field(default_factory=Scoring)
    report: Report = Field(default_factory=Report)

    _source: Path | None = PrivateAttr(default=None)

    @field_validator("models", mode="before")
    @classmethod
    def _names_or_blocks(cls, value: Any) -> Any:
        """models = ["a", "b"] is short for one [[models]] block per name; the two forms can be mixed."""
        if isinstance(value, list):
            return [{"name": item} if isinstance(item, str) else item for item in value]
        return value

    @model_validator(mode="after")
    def _finish(self, info: ValidationInfo) -> Config:
        if not self.endpoints:
            env = (info.context or {}).get("env") if info.context else None
            self.endpoints = [default_endpoint(env)]
        names = [e.name for e in self.endpoints]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"endpoint names must be unique; repeated: {', '.join(dupes)}")
        seen: set[tuple[str, str, str | None]] = set()
        for i, model in enumerate(self.models):
            endpoint = model.endpoint or names[0]
            if endpoint not in names:
                raise ValueError(
                    f"models[{i}] ({model.name}) uses endpoint '{endpoint}', "
                    f"but the endpoints are: {', '.join(names)}"
                )
            if (model.name, endpoint, model.thinking) in seen:
                raise ValueError(
                    f"models[{i}] ({model.label}) is listed twice for endpoint '{endpoint}'; "
                    "a model may be repeated only with a different 'thinking' setting"
                )
            seen.add((model.name, endpoint, model.thinking))
        return self

    @property
    def source(self) -> Path | None:
        return self._source

    @property
    def base_dir(self) -> Path:
        """Relative paths in the config resolve against the config file's folder."""
        return self._source.parent if self._source else Path.cwd()

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(os.path.expandvars(str(value))).expanduser()
        return path if path.is_absolute() else self.base_dir / path

    def endpoint(self, name: str | None = None) -> Endpoint:
        if name is None:
            return self.endpoints[0]
        for endpoint in self.endpoints:
            if endpoint.name == name:
                return endpoint
        raise KeyError(name)

    def enabled_models(self) -> list[ModelEntry]:
        return [m for m in self.models if m.enabled]


# --------------------------------------------------------------------------- loading


def find_config(
    explicit: str | Path | None = None, env: Mapping[str, str] | None = None, cwd: Path | None = None
) -> Path | None:
    """--config, then $GENBENCH_CONFIG, then ./genbench.toml; None means 'use defaults'."""
    env = os.environ if env is None else env
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        return path.resolve()
    if env.get(ENV_CONFIG):
        path = Path(env[ENV_CONFIG]).expanduser()
        if not path.is_file():
            raise ConfigError(f"{ENV_CONFIG} points to a missing file: {path}")
        return path.resolve()
    candidate = (cwd or Path.cwd()) / CONFIG_FILENAME
    return candidate.resolve() if candidate.is_file() else None


def load_config(
    path: str | Path | None = None, overrides: Iterable[str] = (), env: Mapping[str, str] | None = None
) -> Config:
    env = os.environ if env is None else env
    data: dict[str, Any] = {}
    if path is not None:
        try:
            text = Path(path).read_bytes().decode("utf-8-sig")
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{path}: invalid TOML: {exc}") from None
        except (OSError, UnicodeDecodeError) as exc:
            raise ConfigError(f"{path}: can't read it: {exc}") from None
    for parts, value in _env_overrides(env):
        _set_path(data, parts, value, f"environment variable {ENV_PREFIX}{'__'.join(map(str, parts)).upper()}")
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise ConfigError(f"--set needs KEY=VALUE, got {item!r}")
        _set_path(data, _split_key(key.strip()), _parse_value(raw.strip()), f"--set {item}")
    try:
        config = Config.model_validate(data, context={"env": env})
    except ValidationError as exc:
        raise ConfigError(format_validation_error(exc, str(path) if path else "settings")) from None
    config._source = Path(path).resolve() if path else None
    return config


def read_models_file(path: str | Path) -> list[str]:
    """One model per line; '#' comments and pasted `ollama list` output (header included) are fine."""
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ConfigError(f"can't read models file {path}: {exc.strerror or exc}") from None
    models: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lower = line.lower()
        if lower.startswith("name ") and "modified" in lower:
            continue
        name = line.split()[0]
        if name not in models:
            models.append(name)
    return models


def format_validation_error(exc: ValidationError, where: str) -> str:
    lines = [f"{where}: {exc.error_count()} problem(s)"]
    for err in exc.errors(include_url=False):
        loc = format_loc(err["loc"]) or "(top level)"
        lines.append(f"  - {loc}: {clean_message(err['msg'])}")
    return "\n".join(lines)


def format_loc(loc: Iterable[Any]) -> str:
    out = ""
    for part in loc:
        out += f"[{part}]" if isinstance(part, int) else ("." if out else "") + str(part)
    return out


def clean_message(msg: str) -> str:
    return msg.removeprefix("Value error, ")


def _env_overrides(env: Mapping[str, str]) -> Iterator[tuple[list[str | int], Any]]:
    """GENBENCH_RUN__REPEATS=3 sets run.repeats; '__' separates levels."""
    for name in sorted(env):
        if not name.startswith(ENV_PREFIX) or name == ENV_CONFIG or "__" not in name:
            continue
        parts: list[str | int] = []
        for part in name[len(ENV_PREFIX):].split("__"):
            if part:
                parts.append(int(part) if part.isdigit() else part.lower())
        if parts:
            yield parts, _parse_value(env[name])


def _split_key(key: str) -> list[str | int]:
    """run.repeats, scoring.weights."category.coding", endpoints.0.url"""
    parts: list[str | int] = []
    for match in re.finditer(r'"([^"]*)"|([^."]+)', key):
        quoted, bare = match.group(1), match.group(2)
        if quoted is not None:
            parts.append(quoted)
        else:
            parts.append(int(bare) if bare.isdigit() else bare)
    if not parts:
        raise ConfigError(f"empty key in override {key!r}")
    return parts


def _parse_value(raw: str) -> Any:
    """TOML literal if it parses (3, true, [1, 2], "x"), otherwise the raw text."""
    try:
        return tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        return raw


def _set_path(data: dict[str, Any], parts: list[str | int], value: Any, origin: str) -> None:
    node: Any = data
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        if isinstance(part, int):
            if not isinstance(node, list) or part >= len(node):
                raise ConfigError(f"{origin}: there is no item {part} at {format_loc(parts[:i]) or 'top level'}")
            if last:
                node[part] = value
            else:
                node = node[part]
        else:
            if not isinstance(node, dict):
                raise ConfigError(f"{origin}: {format_loc(parts[:i])} is not a table")
            if last:
                node[part] = value
            else:
                node = node.setdefault(part, {})
