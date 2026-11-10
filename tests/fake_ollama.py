"""A small in-process Ollama: just enough of the API for backend and end-to-end tests."""

import json

import httpx

GiB = 1024**3


class FakeOllama:
    def __init__(self, answers=None, models=("m1:latest", "m2:latest"), down_hosts=(), thinking="Let me think.",
                 prompt_tokens=200, context_length=32768, context_lengths=None, completion_tokens=400,
                 done_reason="stop", families=None, model_info=None, refuse_think=()):
        self.answers = answers or {}  # substring of the prompt -> answer text
        self.models = list(models)
        self.down_hosts = set(down_hosts)
        self.thinking = thinking
        self.prompt_tokens = prompt_tokens
        self.context_length = context_length
        self.context_lengths = context_lengths or {}  # model -> context window, when models differ
        self.completion_tokens = completion_tokens
        self.done_reason = done_reason
        self.families = families or {}  # model -> family name, e.g. "qwen35moe"
        self.model_info = model_info or {}  # model -> GGUF metadata
        self.refuse_think = set(refuse_think)  # 'think' values this server rejects, e.g. {"low"}
        self.send_final = True  # False: the stream ends without the closing message and its token counts
        self.requests: list[tuple[str, dict]] = []

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=self.transport())

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.host in self.down_hosts:
            raise httpx.ConnectError("down", request=request)
        body = json.loads(request.content) if request.content else {}
        path = request.url.path
        self.requests.append((path, body))
        if path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": m} for m in self.models]})
        if path == "/api/version":
            return httpx.Response(200, json={"version": "0.12.9"})
        if path == "/api/show":
            name = body.get("model") or ""
            name = name if name in self.models else name + ":latest"
            details = {"quantization_level": "Q4_K_M"}
            if name in self.families:
                details.update(family=self.families[name], families=[self.families[name]])
            return httpx.Response(200, json={"parameters": f"num_ctx {self.context_length}\nstop <eot>",
                                             "details": details, "model_info": self.model_info.get(name, {})})
        if path == "/api/ps":
            return httpx.Response(200, json={"models": [
                {"name": m, "model": m, "size": 20 * GiB, "size_vram": 18 * GiB,
                 "context_length": self.context_lengths.get(m, self.context_length)} for m in self.models]})
        if path == "/api/generate":
            return httpx.Response(200, json={"done": True})
        if path == "/api/chat":
            if body["model"] not in self.models and body["model"] + ":latest" not in self.models:
                return httpx.Response(404, json={"error": f"model '{body['model']}' not found"})
            if "think" in body and body["think"] in self.refuse_think:
                return httpx.Response(400, json={"error": f'invalid think value: "{body["think"]}"'})
            if not body.get("stream"):
                return httpx.Response(200, json={"done": True, "load_duration": 6_170_000_000})
            prompt = body["messages"][-1]["content"]
            answer = next((a for key, a in self.answers.items() if key in prompt), "I don't know.")
            return httpx.Response(200, content=self.stream(answer, thinking=body.get("think") is not False))
        return httpx.Response(404, json={"error": "unknown path"})

    def stream(self, answer: str, thinking: bool = True) -> bytes:
        packets = []
        if self.thinking and thinking:
            packets.append({"message": {"role": "assistant", "content": "", "thinking": self.thinking}, "done": False})
        for i in range(0, len(answer), 40):
            packets.append({"message": {"role": "assistant", "content": answer[i:i + 40]}, "done": False})
        packets.append({
            "message": {"role": "assistant", "content": ""}, "done": True, "load_duration": 0,
            "done_reason": self.done_reason,
            "prompt_eval_count": self.prompt_tokens, "prompt_eval_duration": 500_000_000,
            "eval_count": self.completion_tokens, "eval_duration": 8_000_000_000,
        })
        if not self.send_final:
            packets.pop()
        return "\n".join(json.dumps(p) for p in packets).encode("utf-8") + b"\n"
