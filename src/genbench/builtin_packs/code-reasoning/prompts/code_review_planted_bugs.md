You are reviewing a small Python module before it ships. Every function has a docstring or comment that says what it should do. Find the lines where the code does not do what its own docstring or comment says.

Report only real defects: places where the code behaves differently from what is stated. Do not report style, naming, performance, missing input validation or missing features, and do not report code that is unusual but correct.

```text
 1 | """VRAM scheduler and helpers for the inference gateway."""
 2 | import time
 3 |
 4 | IDLE_AFTER_S = 600      # a model is idle when unused for more than 10 minutes
 5 | HIGH_WATER_PCT = 95     # start unloading above this share of VRAM
 6 | LOW_WATER_PCT = 90      # stop unloading once below this share
 7 |
 8 |
 9 | def vram_pct(used_gb, total_gb):
10 |     """Share of VRAM in use, as a percentage from 0 to 100."""
11 |     return used_gb / total_gb
12 |
13 |
14 | def idle_models(models, now):
15 |     """Models unused for more than IDLE_AFTER_S seconds, largest first."""
16 |     idle = [m for m in models if now - m["last_used"] > IDLE_AFTER_S]
17 |     return sorted(idle, key=lambda m: m["size_gb"])
18 |
19 |
20 | def plan_unloads(models, used_gb, total_gb, now, plan=[]):
21 |     """Builds a new plan on every call: the names of the models to unload, in order. Empty unless usage is
22 |     above HIGH_WATER_PCT. Unloads idle models, largest first, and stops once usage is below LOW_WATER_PCT."""
23 |     if vram_pct(used_gb, total_gb) <= HIGH_WATER_PCT:
24 |         return []
25 |     for model in idle_models(models, now):
26 |         plan.append(model["name"])
27 |         used_gb -= model["size_gb"]
28 |         if vram_pct(used_gb, total_gb) < LOW_WATER_PCT:
29 |             break
30 |     return plan
31 |
32 |
33 | def batches_needed(prompts, batch_size):
34 |     """How many batches of at most batch_size it takes to run all the prompts."""
35 |     return -(-len(prompts) // batch_size)
36 |
37 |
38 | class RateLimiter:
39 |     """Allows at most `limit` requests per client within any `window_s` seconds."""
40 |
41 |     def __init__(self, limit, window_s=60):
42 |         self.limit = limit
43 |         self.window_s = window_s
44 |         self.seen = {}
45 |
46 |     def allow(self, client, now=None):
47 |         now = time.monotonic() if now is None else now
48 |         recent = [t for t in self.seen.get(client, []) if now - t < self.window_s]
49 |         if len(recent) > self.limit:
50 |             return False
51 |         recent.append(now)
52 |         self.seen[client] = recent
53 |         return True
54 |
55 |
56 | def retry_delays(base_s, attempts, cap_s=30):
57 |     """Backoff delays in seconds, one per attempt: base, 2x base, 4x base and so on, each capped at cap_s."""
58 |     return [min(cap_s, base_s * 2 ** n) for n in range(1, attempts)]
```

Reply with ONLY a fenced ```json block containing a JSON array with one object per defect, in this form:
{"line": <line number>, "problem": "<one sentence>", "fix": "<the corrected line>"}
