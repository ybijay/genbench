You are reviewing three artifacts for a self-hosted AI coding platform before it goes live. Find where they disagree.

ARTIFACT 1: Requirements
- R1: Up to 8 requests must be processed at the same time, without queueing.
- R2: Time to first token (TTFT) must stay under 2 seconds for prompts of up to 8,000 tokens.
- R3: Every request must be authenticated with a per-developer API token.
- R4: Prompts and completions must never leave the local network.
- R5: Chat models must run with a context window of at least 32,768 tokens.

ARTIFACT 2: Architecture notes
- A1: One Ollama server on host gpu01 (one 24 GB GPU) handles all inference.
- A2: OLLAMA_NUM_PARALLEL=2: two requests run at once and the rest wait in a FIFO queue.
- A3: The gateway checks a per-developer API token on every request.
- A4: When more than 10 requests are waiting, the gateway sends the overflow to a hosted cloud model API.
- A5: num_ctx is set to 16,384 so that two parallel requests fit in VRAM.
- A6: Gateway logs are kept for 30 days.

ARTIFACT 3: Gateway configuration (gateway.yaml)
```yaml
upstream: http://gpu01:11434                          # C1
auth: none                                            # C2
overflow_provider: https://api.cloud-llm.example/v1   # C3
num_parallel: 2                                       # C4
num_ctx: 16384                                        # C5
ttft_slo_ms: 5000                                     # C6
log_retention_days: 30                                # C7
```

Tasks:
1. List every inconsistency between the artifacts. Compare only what the artifacts state; do not add requirements of your own.
2. Rate each one Critical, Major or Minor.
3. For each, cite the item IDs involved (for example R3, C2), explain the conflict in one sentence, and recommend a specific correction.

Output only one markdown table with exactly these columns: Severity | Items | Conflict | Recommendation
