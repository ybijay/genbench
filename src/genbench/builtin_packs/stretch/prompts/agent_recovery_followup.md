You are the on-call agent for a self-hosted AI coding platform. Developer IDEs send requests to an inference gateway (`llm-gateway`), which forwards them to an Ollama server (`ollama-gpu01`) running on a single 24 GB GPU host (`gpu01`). A retrieval service (`vector-db`) runs alongside them.

Tools:
- get_service_health(service: str) -> JSON {"status": "healthy" | "degraded" | "down", "error_rate_pct", "p95_latency_ms"}
- get_logs(service: str, lines: int) -> plain text log lines
- get_gpu_status(host: str) -> JSON {"vram_used_gb", "vram_total_gb", "loaded_models": [{"name", "size_gb", "idle_s"}]}
- restart_service(service: str)
- unload_model(host: str, model: str)
- create_incident(severity: "sev1" | "sev2" | "sev3", title: str)

Runbook:
1. All calls in one step run in parallel, so no call can use the result of another call in the same step.
2. Retry a call only if its response was missing, cut off, malformed or an error. Never repeat a call whose response was valid.
3. Restart a service only if its own valid health check reports "down".
4. If a valid get_gpu_status response shows more than 95% of VRAM in use, unload idle models (idle_s above 600) one at a time, largest first, and stop as soon as usage would be below 90%. Never unload a model whose idle_s is 600 or less.
5. Open one incident for each service that is "down" or has an error rate above 25%: sev1 if it is down or above 50%, otherwise sev2. Put the service name in the title.

Step 1 made these five calls in parallel:
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

Step 2 retried the two calls that failed:
6. get_service_health(service="ollama-gpu01")
   -> {"status": "degraded", "error_rate_pct": 31.2, "p95_latency_ms": 28700}
7. get_gpu_status(host="gpu01")
   -> {"vram_used_gb": 23.6, "vram_total_gb": 24.0, "loaded_models": [
        {"name": "qwen3.6:35b", "size_gb": 20.5, "idle_s": 2},
        {"name": "nomic-embed-text", "size_gb": 0.6, "idle_s": 1800},
        {"name": "llama3.2-vision:11b", "size_gb": 2.5, "idle_s": 5400}]}

Decide step 3. Reply with ONLY a fenced ```json block containing a JSON array of the tool calls to make now, each in this form:
{"name": "<tool_name>", "arguments": {"<argument>": <value>}}
If no call is needed, reply with an empty array.
