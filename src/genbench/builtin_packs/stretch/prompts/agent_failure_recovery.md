You are the on-call agent for a self-hosted AI coding platform. Developer IDEs send requests to an inference gateway (`llm-gateway`), which forwards them to an Ollama server (`ollama-gpu01`) running on a single 24 GB GPU host (`gpu01`). A retrieval service (`vector-db`) runs alongside them.

Tools (any of them can run in parallel):
- get_service_health(service: str) -> JSON {"status", "error_rate_pct", "p95_latency_ms"}
- get_logs(service: str, lines: int) -> plain text log lines
- get_gpu_status(host: str) -> JSON {"vram_used_gb", "vram_total_gb", "loaded_models"}
- restart_service(service: str)
- unload_model(host: str, model: str)
- create_incident(severity: "sev1" | "sev2" | "sev3", title: str)

Runbook:
1. Retry a call only if its response was missing, cut off, malformed or an error. Never repeat a call whose response was valid.
2. Restart a service only if its own valid health check reports "down". Unload a model only if a valid get_gpu_status response shows more than 95% of VRAM in use.
3. Open an incident only if a service is "down" or its error rate is above 25%.

Your previous step made these five calls in parallel. Here is what came back:

1. get_service_health(service="llm-gateway")
   -> {"status": "degraded", "error_rate_pct": 18.4, "p95_latency_ms": 9400}
2. get_logs(service="llm-gateway", lines=200)
   -> 14:02:11 WARN upstream ollama-gpu01 timed out after 30s (repeated 212 times)
      14:02:12 INFO returned 503 to client after upstream timeout
3. get_service_health(service="ollama-gpu01")
   -> {"status": "healthy", "error_rate_pct": 0.3, "p95_lat
4. get_gpu_status(host="gpu01")
   -> ERROR: request timed out after 10s
5. get_service_health(service="vector-db")
   -> {"status": "healthy", "error_rate_pct": 0.1, "p95_latency_ms": 42}

Decide the next step. Reply with ONLY a fenced ```json block containing a JSON array of the tool calls to make now, each in this form:
{"name": "<tool_name>", "arguments": {"<argument>": <value>}}
If no call is needed, reply with an empty array.
