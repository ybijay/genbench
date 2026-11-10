"""Generated prompt context for long-context tests. The same spec always produces the same text.

The generator uses SplitMix64 instead of Python's random module, so the same seed gives the same text
on any platform and in any language (including the Rust v2 port). Planted lines carry the facts a test
asks about; 'reserved' strings are guaranteed to appear only in planted lines.

Sizing needs no tokenizer: estimate_tokens counts each digit, each run of letters and each other
symbol as one token. Qwen and Gemma split numbers into single digits, so number-heavy logs come out
close to the estimate for them; tokenizers that group digits (Llama 3) come out smaller.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta
from functools import lru_cache

from pydantic import BaseModel, ConfigDict, Field

MASK64 = (1 << 64) - 1


class SplitMix64:
    def __init__(self, seed: int):
        self.state = seed & MASK64

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def below(self, n: int) -> int:
        return self.next() % n

    def pick(self, items: list[str]) -> str:
        return items[self.below(len(items))]


class Plant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    at: float = Field(ge=0, le=1, description="Where the line goes, as a fraction of the log: 0 first, 1 last")
    line: str = Field(min_length=1, description="The line to insert; {time} becomes the timestamp at that point")


class GatewayLogs(BaseModel):
    """An inference-gateway log: requests, scheduler and health lines, with planted lines at chosen depths."""

    model_config = ConfigDict(extra="forbid")
    generator: str = Field("gateway_logs", pattern="^gateway_logs$")
    approx_tokens: int = Field(ge=500, le=500_000, description="Target size in tokens, by estimate_tokens")
    seed: int = Field(ge=0, description="Same seed, same log")
    start: str = Field("2026-11-03 09:00:00", pattern=r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
    plants: list[Plant] = Field(default_factory=list)
    reserved: list[str] = Field(
        default_factory=list, description="Text that may appear only in planted lines (checked every time)"
    )


ContextSpec = GatewayLogs

_TOKEN = re.compile(r"\d|[A-Za-z]+|[^\sA-Za-z\d]")


def estimate_tokens(text: str) -> int:
    """Tokenizer-free size estimate: one token per digit, per run of letters and per other symbol."""
    return len(_TOKEN.findall(text))


_CLIENTS = [f"ide-ws-{i:02d}" for i in range(1, 13)] + ["ci-runner-1", "ci-runner-2", "notebook-lab-3"]
_MODELS = ["qwen3.6:35b", "qwen3.6:35b", "qwen3.6:35b", "nomic-embed-text", "qwen3.8:27b"]
_STATUS = ["200"] * 18 + ["499", "502"]


def _filler(rng: SplitMix64, ts: str) -> str:
    kind = rng.below(100)
    if kind < 82:
        status = rng.pick(_STATUS)
        ms = 300 + rng.below(9000)
        level = "WARN" if status == "502" or ms > 8000 else "INFO"
        return (f"{ts} {level} gateway req=r-{100000 + rng.below(900000)} client={rng.pick(_CLIENTS)} "
                f"model={rng.pick(_MODELS)} status={status} in={50 + rng.below(4000)} out={rng.below(900)} ms={ms}")
    if kind < 90:
        return f"{ts} INFO scheduler queue_depth={rng.below(6)} running={1 + rng.below(2)}"
    if kind < 96:
        return f"{ts} INFO health ollama-gpu01 ok p95_ms={900 + rng.below(1500)}"
    return f"{ts} DEBUG cache prefix_hit tokens={64 * (1 + rng.below(40))}"


def generate(spec: GatewayLogs) -> str:
    """Build the log. Raises ValueError if a reserved string would appear outside the planted lines."""
    return _generate_cached(spec.model_dump_json())


@lru_cache(maxsize=32)
def _generate_cached(spec_json: str) -> str:
    spec = GatewayLogs.model_validate_json(spec_json)
    rng = SplitMix64(spec.seed)
    clock = datetime.strptime(spec.start, "%Y-%m-%d %H:%M:%S")
    target = spec.approx_tokens - sum(estimate_tokens(p.line) + 20 for p in spec.plants)
    lines: list[str] = []
    times: list[str] = []
    size = 0
    while size < target:
        clock += timedelta(seconds=1 + rng.below(9))
        ts = clock.strftime("%Y-%m-%d %H:%M:%S")
        for _ in range(100):
            line = _filler(rng, ts)
            if not any(r in line for r in spec.reserved):
                break
        else:
            raise ValueError("could not generate filler free of the reserved strings")
        lines.append(line)
        times.append(ts)
        size += estimate_tokens(line) + 1
    count = len(lines)
    placed = sorted(((round(p.at * count), i, p) for i, p in enumerate(spec.plants)), reverse=True)
    for index, _, plant in placed:
        stamp = times[min(max(index - 1, 0), count - 1)]
        lines.insert(index, plant.line.replace("{time}", stamp))
    text = "\n".join(lines)
    for reserved in spec.reserved:
        expected = sum(p.line.count(reserved) for p in spec.plants)
        if text.count(reserved) != expected:
            raise ValueError(f"reserved text {reserved!r} appears {text.count(reserved)} times, expected {expected}")
    return text


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def describe(spec: GatewayLogs) -> str:
    text = generate(spec)
    return (f"{spec.generator}, about {estimate_tokens(text):,} tokens ({len(text):,} characters, "
            f"{text.count(chr(10)) + 1:,} lines), seed {spec.seed}, {len(spec.plants)} planted line(s)")
