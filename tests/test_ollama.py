import json

import httpx
import pytest
from fake_ollama import FakeOllama

from genbench.backends.base import BackendError, split_options, timing_metrics
from genbench.backends.ollama import OllamaBackend
from genbench.config import Endpoint


def backend(fake, **endpoint):
    ep = Endpoint(name="gpu", url=endpoint.pop("url", "gpu-box"), **endpoint)
    return OllamaBackend(ep, client=fake.client())


def test_timing_matches_the_original_formulas():
    final = {"load_duration": 1.5e9, "prompt_eval_count": 300, "prompt_eval_duration": 0.2e9,
             "eval_count": 400, "eval_duration": 8e9}
    m = timing_metrics(0.0, 2.0, 10.0, final, "x")
    assert (m["total_s"], m["wall_ttft_s"], m["load_s"], m["warm_ttft_s"]) == (10.0, 2.0, 1.5, 0.5)
    assert (m["tokens_per_sec"], m["prompt_eval_tps"]) == (50.0, 1500.0)
    # Warm TTFT far above prompt-eval time falls back to the prompt-eval time.
    assert timing_metrics(0.0, 5.0, 10.0, {**final, "load_duration": 0}, "x")["warm_ttft_s"] == 0.2
    # No server timings: estimate from characters over the time after the first token.
    estimated = timing_metrics(0.0, 1.0, 5.0, {}, "a" * 400)
    assert estimated["tokens_per_sec"] == 25.0 and estimated["prompt_eval_tps"] == 0.0


def test_options_are_split_for_ollama():
    opts, keep_alive = split_options({"temperature": 0, "keep_alive": "15m", "max_tokens": 512})
    assert opts == {"temperature": 0, "num_predict": 512} and keep_alive == "15m"


def test_connect_uses_the_first_url_that_answers():
    fake = FakeOllama(down_hosts={"blueboy"})
    b = backend(fake, url="blueboy", fallback_urls=["blueboy.local"])
    assert b.connect() == "http://blueboy.local:11434"
    assert b.version == "0.12.9"
    with pytest.raises(BackendError, match="can't reach endpoint 'gpu'"):
        backend(FakeOllama(down_hosts={"gpu-box"})).connect()


def test_warmup_reads_load_time_vram_and_context():
    fake = FakeOllama()
    info = backend(fake).warmup("m1", {"temperature": 0.0, "keep_alive": "15m", "num_ctx": 65536})
    assert info.error is None
    assert info.cold_load_s == 6.17
    assert info.label == "18.0GB (90% GPU @ 32k)"
    chat = next(body for path, body in fake.requests if path == "/api/chat")
    assert chat["options"]["num_predict"] == 1 and chat["options"]["num_ctx"] == 65536 and chat["keep_alive"] == "15m"


def test_warmup_failure_is_reported_not_raised():
    info = backend(FakeOllama()).warmup("missing-model", {})
    assert info.error and info.label.startswith("Warmup error")


def test_chat_streams_content_thinking_and_metrics():
    fake = FakeOllama(answers={"hello": "Hello! " * 20})
    attempt = backend(fake).chat("m1", [{"role": "user", "content": "say hello"}],
                                 {"temperature": 0.0, "keep_alive": "15m", "max_tokens": 64})
    assert attempt.content == "Hello! " * 20
    assert attempt.thinking == "Let me think."
    assert attempt.tokens_per_sec == 50.0 and attempt.prompt_eval_tps == 400.0
    assert attempt.prompt_tokens == 200 and attempt.completion_tokens == 400
    assert attempt.first_thinking_s is not None and attempt.first_content_s is not None
    sent = [body for path, body in fake.requests if path == "/api/chat"][-1]
    assert sent["stream"] is True and sent["keep_alive"] == "15m"
    assert sent["options"] == {"temperature": 0.0, "num_predict": 64}


def test_chat_errors_become_backend_errors():
    with pytest.raises(BackendError, match="HTTP 404: model 'nope' not found"):
        backend(FakeOllama()).chat("nope", [{"role": "user", "content": "x"}], {})

    def broken(request):
        if request.url.path == "/api/chat":
            lines = [json.dumps({"message": {"content": "partial"}, "done": False}), json.dumps({"error": "out of memory"})]
            return httpx.Response(200, content="\n".join(lines).encode())
        return httpx.Response(200, json={"models": []})

    b = OllamaBackend(Endpoint(name="gpu", url="gpu-box"), client=httpx.Client(transport=httpx.MockTransport(broken)))
    with pytest.raises(BackendError, match="out of memory"):
        b.chat("m1", [{"role": "user", "content": "x"}], {})


def test_native_tool_calls_are_collected():
    def tools(request):
        if request.url.path == "/api/chat":
            packet = {"message": {"content": "", "tool_calls": [{"function": {"name": "search", "arguments": {"q": "x"}}}]},
                      "done": True, "eval_count": 5, "eval_duration": 1e8}
            return httpx.Response(200, content=json.dumps(packet).encode())
        return httpx.Response(200, json={"models": []})

    b = OllamaBackend(Endpoint(name="gpu", url="gpu-box"), client=httpx.Client(transport=httpx.MockTransport(tools)))
    attempt = b.chat("m1", [{"role": "user", "content": "find x"}], {}, tools=[{"type": "function"}])
    assert attempt.tool_calls == [{"function": {"name": "search", "arguments": {"q": "x"}}}]
