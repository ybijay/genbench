"""The stretch pack: good and bad sample answers must score as intended."""

import json
from pathlib import Path

import pytest

from genbench.config import Config
from genbench.grade.engine import GradeContext, grade_test
from genbench.grade.validators import ValidatorRunner
from genbench.pack.loader import BUILTIN_DIR, Catalog, LoadedTest, load_pack
from genbench.pack.validate import validate_catalog
from genbench.schema import pack_schema

PACK = BUILTIN_DIR / "stretch" / "pack.json"
pytestmark = pytest.mark.skipif(not PACK.is_file(), reason="the stretch pack is not present")


@pytest.fixture(scope="module")
def pack():
    loaded, issues = load_pack(PACK)
    assert not issues, issues
    return loaded


def grade(pack, test_id, answer, tmp_path):
    test = next(t for t in pack.pack.tests if t.id == test_id)
    cfg = Config()
    ctx = GradeContext(ValidatorRunner(cfg.validators), tmp_path, pack.dir, cfg.scoring.pass_mark)
    result = grade_test(test, answer, ctx)
    return result.accuracy, result.quality, result.passed


def test_pack_is_valid_and_portable(pack):
    cfg = Config.model_validate({"packs": {"builtin": False, "dirs": [str(PACK.parent)]}})
    issues = validate_catalog(Catalog(packs=[pack]), cfg, portable=True)
    assert [i for i in issues if i.level == "error" or "Rust" in i.message] == []
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator(pack_schema()).validate(json.loads(PACK.read_text(encoding="utf-8")))
    assert [t.id for t in pack.pack.tests] == ["agent_failure_recovery", "artifact_consistency_review",
                                            "agent_recovery_followup", "artifact_review_subtle"]
    assert all(LoadedTest(t, pack).prompt_text().count("\n") > 10 for t in pack.pack.tests)


# --------------------------------------------------------------------------- agent_failure_recovery

RETRY_HEALTH = {"name": "get_service_health", "arguments": {"service": "ollama-gpu01"}}
RETRY_GPU = {"name": "get_gpu_status", "arguments": {"host": "gpu01"}}


def fenced(calls) -> str:
    return "```json\n" + json.dumps(calls, indent=2) + "\n```"


AGENT_CASES = [
    ("perfect", fenced([RETRY_HEALTH, RETRY_GPU]), (100.0, 100.0, True)),
    ("perfect-after-thinking", "<think>3 and 4 are broken</think>\n" + fenced([RETRY_GPU, RETRY_HEALTH]), (100.0, 100.0, True)),
    ("extra-args-are-fine", fenced([{"name": "get_service_health", "arguments": {"service": "ollama-gpu01", "x": 1}}, RETRY_GPU]),
     (100.0, 100.0, True)),
    ("prose-around-json", "Calls 3 and 4 failed, so:\n\n" + fenced([RETRY_HEALTH, RETRY_GPU]), (100.0, 50.0, True)),
    ("wrapped-in-object", "```json\n" + json.dumps({"tool_calls": [RETRY_HEALTH, RETRY_GPU]}) + "\n```", (100.0, 50.0, True)),
    ("restarts-gateway", fenced([RETRY_HEALTH, RETRY_GPU, {"name": "restart_service", "arguments": {"service": "llm-gateway"}}]),
     (85.0, 100.0, False)),
    ("opens-incident", fenced([RETRY_HEALTH, RETRY_GPU, {"name": "create_incident", "arguments": {"severity": "sev2", "title": "x"}}]),
     (85.0, 100.0, False)),
    ("retries-valid-logs", fenced([RETRY_HEALTH, RETRY_GPU, {"name": "get_logs", "arguments": {"service": "llm-gateway", "lines": 200}}]),
     (80.0, 100.0, False)),
    ("trusts-truncated-health", fenced([RETRY_GPU]), (75.0, 100.0, False)),
    ("does-nothing", fenced([]), (50.0, 100.0, False)),
    ("unknown-tool", fenced([RETRY_HEALTH, {"name": "check_gpu", "arguments": {"host": "gpu01"}}]), (75.0, 50.0, False)),
    ("no-json", "I would retry the ollama-gpu01 health check and the GPU status.", (0.0, 0.0, False)),
]


@pytest.mark.parametrize(("answer", "expected"), [c[1:] for c in AGENT_CASES], ids=[c[0] for c in AGENT_CASES])
def test_agent_failure_recovery(pack, answer, expected, tmp_path):
    assert grade(pack, "agent_failure_recovery", answer, tmp_path) == expected


# --------------------------------------------------------------------------- artifact_consistency_review

HEADER = "| Severity | Items | Conflict | Recommendation |\n|---|---|---|---|"
ROWS = {
    "auth": "| Critical | R3, A3, C2 | C2 sets auth: none, so requests are not authenticated although R3 and A3 require per-developer tokens. | Enable per-developer API token checks in gateway.yaml. |",
    "cloud": "| Critical | R4, A4, C3 | Overflow requests go to a hosted cloud API, so prompts leave the local network, which R4 forbids. | Remove the cloud overflow provider and queue or reject excess requests locally. |",
    "parallel": "| Major | R1, A2, C4 | Only 2 requests run at once and the rest are queued, but R1 requires 8 at the same time without queueing. | Provide capacity for 8 concurrent requests or revise R1. |",
    "context": "| Major | R5, A5, C5 | num_ctx is 16,384, below the 32,768-token context R5 requires. | Set num_ctx to at least 32,768 and size VRAM for it. |",
    "ttft": "| Major | R2, C6 | ttft_slo_ms is 5000 ms, but R2 requires TTFT under 2 seconds. | Set ttft_slo_ms to 2000. |",
}
RETENTION = "| Minor | A6, C7 | Logs are kept for 30 days but no retention requirement exists. | Add a retention requirement. |"
TLS = "| Minor | A1 | The upstream uses plain HTTP without TLS. | Use HTTPS internally. |"
SPLIT = "| Critical | R3, C2 | The config disables auth entirely. | Turn on token auth. |"


def table(*rows: str, header: str = HEADER) -> str:
    return header + "\n" + "\n".join(rows)


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "stretch"
NO_SEPARATOR = HEADER.splitlines()[0] + "\n" + "\n".join(ROWS.values())
SPACED = table(*[row.replace("5000 ms", "5 000 ms") for row in ROWS.values()]).replace("16,384", "16 384").replace("32,768", "32 768")

REVIEW_CASES = [
    ("perfect", table(*ROWS.values()), (100.0, 100.0, True)),
    ("conflicts-split-per-artifact", table(*ROWS.values(), SPLIT), (100.0, 100.0, True)),
    ("misses-ttft", table(ROWS["auth"], ROWS["cloud"], ROWS["parallel"], ROWS["context"], SPLIT), (85.0, 100.0, False)),
    ("invents-retention-issue", table(*ROWS.values(), RETENTION), (85.0, 100.0, False)),
    ("invents-tls-issue", table(*ROWS.values(), TLS), (85.0, 100.0, False)),
    ("auth-only-major", table(ROWS["auth"].replace("| Critical |", "| Major |", 1), *list(ROWS.values())[1:]), (100.0, 80.0, True)),
    ("wrong-columns", table(*ROWS.values(), header="| Issue | Severity | Fix |\n|---|---|---|"), (100.0, 80.0, True)),
    ("padded-with-ten-rows", table(*ROWS.values(), *[SPLIT] * 5), (90.0, 100.0, False)),
    ("prose-not-table", "Auth is disabled in C2, overflow goes to the cloud (A4), and num_ctx is only 16,384.", (0.0, 0.0, False)),
    ("no-separator-row", NO_SEPARATOR, (100.0, 80.0, True)),
    ("spaced-thousands", SPACED, (100.0, 100.0, True)),
    # Real answers from run 20261109-183025 that version 1.0.0 of this pack graded wrongly.
    ("real-muse-glimmer-no-separator", (FIXTURES / "review_muse-glimmer_no_separator.md").read_text(encoding="utf-8"),
     (100.0, 80.0, True)),
    ("real-qwen3.8-64k-spaced-numbers", (FIXTURES / "review_qwen3.8-64k_spaced_numbers.md").read_text(encoding="utf-8"),
     (100.0, 100.0, True)),
]


@pytest.mark.parametrize(("answer", "expected"), [c[1:] for c in REVIEW_CASES], ids=[c[0] for c in REVIEW_CASES])
def test_artifact_consistency_review(pack, answer, expected, tmp_path):
    assert grade(pack, "artifact_consistency_review", answer, tmp_path) == expected


# --------------------------------------------------------------------------- agent_recovery_followup

UNLOAD_VISION = {"name": "unload_model", "arguments": {"host": "gpu01", "model": "llama3.2-vision:11b"}}
UNLOAD_EMBED = {"name": "unload_model", "arguments": {"host": "gpu01", "model": "nomic-embed-text"}}
UNLOAD_CODER = {"name": "unload_model", "arguments": {"host": "gpu01", "model": "qwen3.6:35b"}}
SEV2 = {"name": "create_incident", "arguments": {"severity": "sev2", "title": "ollama-gpu01 degraded: 31.2% errors"}}


def incident(severity, title):
    return {"name": "create_incident", "arguments": {"severity": severity, "title": title}}


FOLLOWUP_CASES = [
    ("perfect", fenced([UNLOAD_VISION, SEV2]), (100.0, 100.0, True)),
    ("perfect-other-order", "<think>23.6 - 2.5 = 21.1, below 21.6</think>\n" + fenced([SEV2, UNLOAD_VISION]),
     (100.0, 100.0, True)),
    ("unloads-every-idle-model", fenced([UNLOAD_VISION, UNLOAD_EMBED, SEV2]), (85.0, 100.0, False)),
    ("smallest-first", fenced([UNLOAD_EMBED, UNLOAD_VISION, SEV2]), (85.0, 100.0, False)),
    ("unloads-the-busy-model", fenced([UNLOAD_CODER, SEV2]), (60.0, 100.0, False)),
    ("model-name-without-tag", fenced([{"name": "unload_model", "arguments": {"host": "gpu01", "model": "llama3.2-vision"}},
                                       SEV2]), (75.0, 100.0, False)),
    ("sev1", fenced([UNLOAD_VISION, incident("sev1", "ollama-gpu01 degraded")]), (65.0, 100.0, False)),
    ("title-without-service", fenced([UNLOAD_VISION, incident("sev2", "Inference errors at 31%")]), (90.0, 100.0, False)),
    ("gateway-incident-too", fenced([UNLOAD_VISION, SEV2, incident("sev3", "llm-gateway degraded")]),
     (85.0, 100.0, False)),
    ("rechecks-the-gpu", fenced([UNLOAD_VISION, SEV2, {"name": "get_gpu_status", "arguments": {"host": "gpu01"}}]),
     (85.0, 100.0, False)),
    ("restarts-ollama", fenced([UNLOAD_VISION, SEV2, {"name": "restart_service", "arguments": {"service": "ollama-gpu01"}}]),
     (85.0, 100.0, False)),
    ("incident-only", fenced([SEV2]), (75.0, 100.0, False)),
    ("empty-array", fenced([]), (45.0, 100.0, False)),
    ("prose-around-json", "Unload the idle vision model and raise a sev2:\n\n" + fenced([UNLOAD_VISION, SEV2]),
     (100.0, 50.0, True)),
]


@pytest.mark.parametrize(("answer", "expected"), [c[1:] for c in FOLLOWUP_CASES], ids=[c[0] for c in FOLLOWUP_CASES])
def test_agent_recovery_followup(pack, answer, expected, tmp_path):
    assert grade(pack, "agent_recovery_followup", answer, tmp_path) == expected


# --------------------------------------------------------------------------- artifact_review_subtle

SUBTLE = {
    "model": "| Major | C1, R7 | The config uses qwen3.6:latest instead of the pinned qwen3.6:35b-a3b-coding-mtp-q4_K_M. | Set model to the pinned tag. |",
    "rate": "| Major | C2, R2 | The config allows 600 requests a minute per developer, ten times the limit of 60. | Set the limit to 60. |",
    "ttl": "| Major | C3, R4 | token_ttl_hours 8760 is 365 days, far beyond the 90-day maximum. | Set token_ttl_hours to 2160. |",
    "body": "| Major | C4, R5, A1 | max_body_bytes 2000000 rejects bodies between 2,000,000 and 2,097,152 bytes. | Set it to 2097152. |",
    "ttft": "| Major | A5, C6, R1 | The dashboard and config target average TTFT, but the requirement is on p95. | Track p95 TTFT under 2000 ms. |",
    "storage": "| Critical | C7, A3, R3 | log_prompts: true writes prompts and completions to /var/log/gateway on disk. | Set log_prompts to false. |",
}
OVERFLOW_ROW = "| Critical | A2, C9, R8 | Overflow traffic goes to a second server and may leave the network. | Remove the overflow. |"
HTTP_ROW = "| Minor | C9 | The overflow server is reached over plain HTTP without TLS. | Use HTTPS. |"
HEALTH_ROW = "| Minor | C5, R6 | health_interval_ms is 30000 but R6 asks for 30 seconds. | Express it in seconds. |"
VERSION_ROW = "| Minor | A6 | /api/version doesn't prove a model can serve requests. | Use a real inference probe. |"


def subtle(*keys: str, extra: tuple[str, ...] = (), swap: dict[str, str] | None = None, header: str = HEADER) -> str:
    rows = [(swap or {}).get(k, SUBTLE[k]) for k in keys or SUBTLE]
    return table(*rows, *extra, header=header)


ALL = tuple(SUBTLE)
SUBTLE_CASES = [
    ("perfect", subtle(), (100.0, 100.0, True)),
    ("storage-split-per-artifact", subtle(*ALL[:5], extra=(
        "| Critical | C7, R3 | log_prompts: true stores prompts on disk. | Set log_prompts to false. |",
        "| Major | A3, R3 | Gateway logs written to disk would hold prompts and completions. | Keep prompts out of logs. |")),
     (100.0, 100.0, True)),
    ("flags-the-lan-overflow", subtle(extra=(OVERFLOW_ROW,)), (82.0, 100.0, False)),
    ("flags-plain-http", subtle(extra=(HTTP_ROW,)), (82.0, 100.0, False)),
    ("flags-the-health-interval", subtle(extra=(HEALTH_ROW,)), (82.0, 100.0, False)),
    ("flags-api-version", subtle(extra=(VERSION_ROW,)), (82.0, 100.0, False)),
    ("misses-average-vs-p95", subtle(*[k for k in ALL if k != "ttft"]), (78.0, 100.0, False)),
    ("misses-mib-vs-mb", subtle(*[k for k in ALL if k != "body"]), (78.0, 100.0, False)),
    ("misses-hours-vs-days", subtle(*[k for k in ALL if k != "ttl"]), (78.0, 90.0, False)),
    ("storage-only-major", subtle(swap={"storage": SUBTLE["storage"].replace("Critical", "Major")}), (100.0, 80.0, True)),
    ("rate-limit-minor", subtle(swap={"rate": SUBTLE["rate"].replace("Major", "Minor")}), (100.0, 90.0, True)),
    ("no-separator-row", HEADER.splitlines()[0] + "\n" + "\n".join(SUBTLE.values()), (100.0, 80.0, True)),
    ("prose-not-table", "C1 uses latest, C2 allows 600, C7 logs prompts.", (0.0, 0.0, False)),
]


@pytest.mark.parametrize(("answer", "expected"), [c[1:] for c in SUBTLE_CASES], ids=[c[0] for c in SUBTLE_CASES])
def test_artifact_review_subtle(pack, answer, expected, tmp_path):
    assert grade(pack, "artifact_review_subtle", answer, tmp_path) == expected
