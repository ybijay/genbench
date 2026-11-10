You are reviewing three artifacts for the inference gateway of a self-hosted AI coding platform before it goes live. Find where they disagree.

ARTIFACT 1: Requirements
- R1: The 95th-percentile (p95) time to first token must stay under 2 seconds.
- R2: Each developer is limited to 60 requests per minute.
- R3: Prompts and completions must never be written to disk.
- R4: API tokens must expire within 90 days of being issued.
- R5: Request bodies of up to 2 MiB (2,097,152 bytes) must be accepted.
- R6: The gateway must check upstream health every 30 seconds.
- R7: Coding requests must use the pinned model qwen3.6:35b-a3b-coding-mtp-q4_K_M.
- R8: Prompts and completions must never leave the local network.

ARTIFACT 2: Architecture notes
- A1: The gateway accepts request bodies of up to 2 MiB and rejects anything larger with HTTP 413.
- A2: When the main server is busy, the gateway sends the overflow to a second Ollama server inside the LAN (llm.internal.lan).
- A3: The gateway writes its logs to /var/log/gateway and rotates them daily.
- A4: Per-developer API tokens are issued by the company identity service.
- A5: The operations dashboard tracks average TTFT to show that the latency target is met.
- A6: Health checks call /api/version on each Ollama server.

ARTIFACT 3: Gateway configuration (gateway.yaml)
```yaml
model: qwen3.6:latest                              # C1
rate_limit_per_developer_per_minute: 600           # C2
token_ttl_hours: 8760                              # C3
max_body_bytes: 2000000                            # C4
health_interval_ms: 30000                          # C5
ttft_target_avg_ms: 1500                           # C6
log_prompts: true                                  # C7
log_retention_days: 30                             # C8
overflow_provider: http://llm.internal.lan:11434   # C9
```

Tasks:
1. List every inconsistency between the artifacts. Check units carefully. Compare only what the artifacts state: do not add requirements of your own, and do not list items that agree.
2. Rate each one Critical, Major or Minor.
3. For each, cite the item IDs involved (for example R4, A1), explain the conflict in one sentence, and recommend a specific correction.

Output only one markdown table with exactly these columns: Severity | Items | Conflict | Recommendation
