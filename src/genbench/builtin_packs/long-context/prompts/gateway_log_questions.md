You are on call for our self-hosted inference gateway. It fronts an Ollama server and logs every request, scheduler sample, health check, config change and model load. This morning's gateway log is below, between the <<<LOG and LOG>>> lines. The facts you need are spread through it, so read all of it.

<<<LOG
{{context}}
LOG>>>

Answer from the log alone:
1. num_parallel: the value of the scheduler's num_parallel setting at the end of the log.
2. changed_by: who made the most recent change to num_parallel that took effect.
3. oom_model: the model the loader gave up on because the GPU ran out of memory.
4. rate_limited_count: how many requests from client ide-hanuman-07 were rejected with status=429.

Reply with ONLY a fenced ```json block holding one object with exactly these four keys:
{"num_parallel": <integer>, "changed_by": "<user>", "oom_model": "<model name>", "rate_limited_count": <integer>}
