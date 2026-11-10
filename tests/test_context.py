"""The long-context generator: reproducible, sized as asked, plants where asked, reserved text only in plants."""

import pytest
from pydantic import ValidationError

from genbench.context import GatewayLogs, SplitMix64, describe, estimate_tokens, fingerprint, generate

PLANTS = [
    {"at": 0.0, "line": "{time} INFO config first by ops-a"},
    {"at": 0.5, "line": "{time} INFO config middle by ops-b"},
    {"at": 1.0, "line": "{time} INFO config last by ops-c"},
]


def spec(**kw) -> GatewayLogs:
    data = {"approx_tokens": 3000, "seed": 7, "plants": PLANTS, "reserved": ["config ", "ops-"]}
    data.update(kw)
    return GatewayLogs.model_validate(data)


def test_splitmix64_matches_the_reference_sequence():
    rng = SplitMix64(0)  # first outputs of Vigna's splitmix64.c with x = 0, so a Rust port can check itself
    assert [rng.next() for _ in range(3)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]
    assert SplitMix64(1601).below(10) == SplitMix64(1601).below(10)


def test_same_spec_same_text_and_seed_changes_it():
    assert generate(spec()) == generate(spec())
    assert fingerprint(generate(spec())) == fingerprint(generate(spec()))
    assert generate(spec(seed=8)) != generate(spec())


def test_size_follows_the_estimate():
    for tokens in (1000, 8000, 30000):
        text = generate(spec(approx_tokens=tokens, plants=[], reserved=[]))
        assert abs(estimate_tokens(text) - tokens) / tokens < 0.03


def test_estimate_counts_digits_words_and_symbols():
    assert estimate_tokens("2026-11-03") == 10
    assert estimate_tokens("status=200 ok") == 6
    assert estimate_tokens("") == 0


def test_plants_land_at_their_depth_with_a_matching_timestamp():
    lines = generate(spec()).splitlines()
    where = {line.split(" by ")[1]: i for i, line in enumerate(lines) if " config " in line}
    assert where["ops-a"] == 0 and where["ops-c"] == len(lines) - 1
    assert abs(where["ops-b"] / len(lines) - 0.5) < 0.02
    stamps = [line[:19] for line in lines]
    assert stamps == sorted(stamps)  # planted lines keep the log in time order


def test_reserved_text_appears_only_in_plants():
    text = generate(spec())
    assert text.count("config ") == 3 and text.count("ops-") == 3
    assert "ide-hanuman" not in generate(spec(plants=[], reserved=["ide-hanuman"]))


def test_reserved_text_the_filler_always_uses_is_an_error():
    with pytest.raises(ValueError, match="reserved"):
        generate(spec(reserved=["2026-11-03"]))


def test_bad_specs_are_rejected():
    with pytest.raises(ValidationError):
        spec(approx_tokens=100)
    with pytest.raises(ValidationError):
        spec(generator="lorem_ipsum")
    with pytest.raises(ValidationError):
        spec(plants=[{"at": 1.5, "line": "x"}])
    with pytest.raises(ValidationError):
        spec(colour="blue")


def test_describe_reports_the_real_size():
    text = describe(spec())
    assert text.startswith("gateway_logs, about ") and "seed 7, 3 planted line(s)" in text
