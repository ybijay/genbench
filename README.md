# genbench

A configurable benchmark for local LLMs. It is the successor to `../ollama_local_benchmark.py`: the same
tests and scores, but everything that was hardcoded now lives in files. Run settings live in
`genbench.toml`; tests live in JSON packs that anyone can add to without touching the code.

## Quick start on one machine

With Ollama running on the same machine, no config file is needed:

```powershell
uv sync
uv run genbench run                     # shows the models Ollama has installed and asks before running them
uv run genbench run -m llama3.2:3b      # or name the model yourself
```

To fix the list of models, put one line in a `genbench.toml`:

```toml
models = ["llama3.2:3b", "qwen3:8b"]
```

`genbench.example.toml` shows the few other lines people usually need, such as a server on another machine.

**Status: Phase 1.** genbench runs the benchmark end to end against Ollama: it warms up each model, streams
every test, grades it, and writes a run folder with results and reports. Re-grading the old script's 40 saved
results reproduces its scores (40/40 within 0.1) and both saved leaderboards exactly. OpenAI-compatible servers
(LM Studio, llama.cpp, vLLM) arrive in Phase 2.

## Install

Needs Python 3.10 or newer.

With [uv](https://docs.astral.sh/uv/) (recommended; it also installs Python if you don't have it):

```powershell
cd genbench
uv sync                      # creates .venv with genbench and its dependencies
uv run genbench validate     # run any command through uv
```

To get a `genbench` command everywhere: `uv tool install --editable .`

With pip:

```powershell
cd genbench
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -e .
genbench validate
```

## Commands

| Command | What it does |
| --- | --- |
| `genbench init [DIR]` | Write a short `genbench.toml` (only what differs from the defaults), plus `packs/` and `schemas/`. `--host` (repeatable), `--import-models models.txt`, `--no-probe`, `--force`; `--full` writes every section |
| `genbench validate` | Check the config and every pack without calling a model. `--online` also contacts each server and checks the models exist; `--portable` warns about regex features Rust v2 won't support; `--strict` fails on warnings |
| `genbench list tests` | List tests (`--suite`, `--tag`); also `list packs`, `list models`, `list endpoints`, `list runs` |
| `genbench explain TEST` | Show what a test asks, its goal, pass mark and every expectation with its points |
| `genbench new-test` | Start a test from plain-English sentences, then attach a check to each expectation |
| `genbench run` | Run the tests on every enabled model. `--model` and `--test` (repeatable) narrow it down; also `--suite`, `--tag`, `--repeats`. With no models in the config it offers the installed ones (`--all-models` skips the question). Ctrl+C stops cleanly and still writes reports |
| `genbench report RUN` | Rebuild a run folder's reports from its `results.jsonl`, e.g. after changing weights |
| `genbench regrade RUN` | Re-score a run's saved answers with today's packs and config, into a new run folder |
| `genbench regrade --from-python-csv FILE` | Re-score results saved by `ollama_local_benchmark.py` and compare with the old scores |
| `genbench clean` | Delete run folders. With no options: only incomplete ones (interrupted, failed, empty, killed). Also `--all`, `--older-than 30d`, `--keep N` (newest completed), or name folders. `--dry-run` previews; it asks before deleting unless `--yes` |
| `genbench schema` | Print the JSON Schema for packs (`--out FILE` to write it) |

Every command takes `--config FILE` and `--set KEY=VALUE` (repeatable) before the command name, e.g.
`genbench --set run.repeats=3 validate`.

## Running a benchmark

```powershell
uv run genbench validate --online     # check the config, packs, server and models first
uv run genbench run                   # every enabled model x every test
uv run genbench run -m qwen3.6:35b-a3b-coding-mtp-q4_K_M -t bash_network_monitor
```

Each run gets its own folder under `runs/` (from `[run].output_dir`), named by date, time and `[run].name`:

| File | Holds |
| --- | --- |
| `results.jsonl` | One line per attempt: timings, the full answer and thinking, every check's result. Everything else is built from it |
| `leaderboard.md` | Ranked table (Overall, Passed, Accuracy, each category, TTFT, tok/s, VRAM), category winners, and for every test what was asked, its goal and each model's missed expectations |
| `results.csv` | The old script's columns first, then Test ID, Result (PASS, FAIL, SKIPPED or ERROR), Pass Mark, Repeat, Wall TTFT, Error, Quality |
| `summary.json` | Per-model aggregates for scripts and dashboards |
| `run.json` | When it ran, the models and endpoints used, and each test's goal and definition hash |
| `artifacts/` | Extracted code per model and test (e.g. `app.py`, `monitor.sh`) |

`run.json` records each run's status: `running`, `completed`, `interrupted` (Ctrl+C) or `failed`. A run left
`running` for 30 minutes without progress shows as `stopped`. `genbench list runs` shows them all, and
`genbench clean` removes the incomplete ones. It only deletes folders genbench created, and never a run that
is still going.

At warm-up genbench warns when the model doesn't fit fully in VRAM, or when other models are loaded on the
server: both slow a run down and make its speeds unfair to compare.

Each model's block in the console starts with its type and ends with a total:

```text
[1/5] qwen3.6:35b-a3b-coding-mtp-q4_K_M on blueboy  (MoE)
  Warming up... ready (cold load 52.91s | VRAM 20.5GB (100% GPU @ 32k))
  Code Reasoning: Predict a Script's Output                  PASS  92.14s | TTFT 1.29s | 90.44 tok/s | acc 100.0 | quality 100.0
  ...
  Total time 404.84s (6m 45s) | average 84.51 tok/s | 3 tests run, 1 skipped
```

The type is MoE (mixture of experts: only a few experts run per token, so it generates much faster than
its size suggests) or Dense. genbench reads it from the server: an expert count in the model's metadata,
`moe` in its family name, or an active-parameter tag such as `a3b` in its name. Set `type = "moe"` or
`type = "dense"` on a `[[models]]` block to override it. The total is the time the model's tests took, not
counting the model load, and the average matches the Tokens/Sec column of the leaderboard.

Scores work as before: accuracy (did it do what was asked) and quality (how well) per test, both 0-100,
plus PASS or FAIL against the pass mark. Overall weighs accuracy, each category and speed by
`[scoring.weights]`, exactly like the old leaderboard.

**Category columns.** Each test adds one number to its category column (Coding, Architecture, Agentic,
Long Context): by default the average of its accuracy and quality, so a wrong answer lowers the column
even when the format was perfect. A test can choose one of the two with `"category_score": "quality"` or
`"accuracy"` in its rubric; the core pack uses `quality`, which is what the original script did. Harder
tests count for more in Accuracy and the category columns (`[scoring].difficulty_weights`: medium 1,
hard 1.5).

**Tests that don't fit.** Before sending a prompt, genbench checks that it fits the model's context
window with 2,000 tokens to spare for the answer. If not, the test is skipped for that model (`SKIP` in
the console, `SKIPPED` in the reports) instead of failing on a prompt the server would have cut. A test
that any model skipped is left out of every model's scores, so models are always compared on the same
tests; its results are still listed under "Results by test".

A model can also fill its window while thinking and never reach an answer. That attempt fails, and the
console, `results.csv` and the leaderboard say why ("Ran out of context window: 16,269 prompt + 16,499
generated tokens filled the 32,768-token window"), instead of only "Empty response".

**Running a few tests.** When a run has no tests in a category that carries weight in Overall (no
architecture tests, say), that weight is left out and the others are scaled up to 100%, and the
leaderboard says so. Compare Overall only between runs of the same tests.

## Configuration

genbench looks for its config in this order: `--config`, then `$GENBENCH_CONFIG`, then `./genbench.toml`.
With none of those, it uses defaults and talks to the Ollama server in `$OLLAMA_HOST`, or `localhost:11434`.

Every section is optional, and a config only needs what differs from the defaults. This is a complete
config for a benchmark machine that calls Ollama on another one:

```toml
models = ["qwen3:8b", "gemma3:12b"]

[[endpoints]]
name = "gpu-box"
url = "http://gpu-box:11434"
```

`models` is a list of names. To give one model its own settings, write a block in place of its name,
`{ name = "qwen3:8b", options = { num_ctx = 16384 } }`, or use a `[[models]]` section per model. In TOML,
the `models = [...]` line has to come before the first `[section]`.

**Thinking.** A model block can carry `thinking = "off"`, `"on"`, `"low"`, `"medium"` or `"high"`, which
genbench passes to the server with every request. Leave it out and the model uses its own default. A model
may be listed once per setting, so one run compares them side by side:

```toml
models = [
  "ornith-1.5:35b",
  { name = "ornith-1.5:35b", thinking = "low" },
  { name = "ornith-1.5:35b", thinking = "off" },
]
```

The results name them `ornith-1.5:35b`, `ornith-1.5:35b (thinking low)` and `ornith-1.5:35b (thinking off)`,
and the model stays loaded between its settings. Not every model accepts every value (many take only on
and off). When the server refuses one, genbench says so once at warm-up, marks that entry's tests as errors
and carries on with the rest.

Settings are layered, lowest to highest:

1. Built-in defaults
2. `genbench.toml`
3. Environment variables `GENBENCH_<SECTION>__<KEY>`, e.g. `GENBENCH_RUN__REPEATS=3`
4. `--set` flags, e.g. `--set endpoints.0.url=http://192.168.1.50:11434` or
   `--set 'scoring.weights."category.coding"=0.3'`

The main sections:

| Section | Holds |
| --- | --- |
| `[run]` | Run name, output folder, repeats, seed |
| `[[endpoints]]` | One block per server: `name`, `kind` (`ollama` or `openai`), `url`, `fallback_urls`, `timeout_s`, `api_key_env` |
| `models` | A list of model names, or blocks with `name`, `endpoint`, per-model `options`, `enabled`, `type` (`moe` or `dense`; read from the server when left out), `thinking`. With none listed, `genbench run` offers the installed ones |
| `[defaults.options]` | Sampling options sent with every request (`temperature`, `keep_alive`, `num_ctx`, ...) |
| `[packs]` | `builtin` (load the tests that ship with genbench), `dirs` (your pack folders; `packs/` by default), `suite`, tag and id filters |
| `[validators]` | Tools that syntax-check generated code: `python = { builtin = "ast" }`, `bash = { cmd = [...] }` (by default `bash`, or Git Bash on Windows when it is installed); `on_missing` says what happens when a tool isn't installed |
| `[scoring]` | `pass_mark`, difficulty weights, and `[scoring.weights]` for the Overall score |
| `[report]` | Output formats and leaderboard title |

**Running somewhere else.** Nothing about Hanuman or Blueboy is built in. On another machine, run
`genbench init --host my-gpu-box` (or `--host` several times, or none for localhost) and review the file it
writes.

**A small machine is enough.** genbench only sends prompts to the model server and grades the text that
comes back, so it needs no GPU and little memory; an old laptop can drive a run while the server does the
work. Speeds and prompt times come from the server's own counters, so they don't depend on the machine
genbench runs on. To set one up:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # installs uv, which brings its own Python
git clone <your repository> genbench && cd genbench
uv sync
cp genbench.example.toml genbench.toml              # then point it at your server and list your models
uv run genbench validate --online
tmux new -s bench 'uv run genbench run'             # tmux keeps the run going if your connection drops
```

A run takes hours, so stop the machine from sleeping while it runs.

## Test packs

A pack is a folder with a `pack.json`. Four packs with 14 tests ship with genbench, in
`src/genbench/builtin_packs`: `core` (the four original tasks), `stretch`, `long-context` and
`code-reasoning`. Put your own under `packs/`; that folder is loaded automatically.

`stretch` holds four harder tests, written because every model scores 100% on the core pack:

- `agent_failure_recovery`: an on-call agent sees five tool results (one cut off, one timed out) and must
  retry only those two, without repeating good calls, restarting, unloading or escalating against a runbook.
- `agent_recovery_followup`: the next step. The retries came back valid (31.2% errors, VRAM at 98%), and a
  five-rule runbook decides what to do: unload only the largest idle model, open one sev2 incident, repeat
  nothing.
- `artifact_consistency_review`: requirements, architecture notes and a gateway config with five real
  conflicts and two traps; the answer is a markdown table, and invented issues fail the test.
- `artifact_review_subtle`: the same kind of review, but the six conflicts hide in units (MB vs MiB,
  hours vs days), metrics (average vs p95) and a floating model tag, next to look-alikes that agree
  once converted (30,000 ms vs 30 s).

`long-context` asks questions about a generated inference-gateway log:

- `long_context_16k`: four facts planted at different depths of a 16k-token log.
- `long_context_24k`: the same questions over 24k tokens with decoys (a look-alike client, a model that ran
  out of memory but loaded on retry, a rejected config change).
- `long_context_tally_12k`: six questions that each need every line: two counts, a total, the slowest
  request, the worst health check, and one the log can't answer (the right answer is `null`). Five of six
  passes. The log is 12k tokens so a model with a 32k window keeps about 20k to work through it; on a 16k
  log one model filled its window re-typing the lines before it could answer.
- `long_context_48k`: the 24k questions over 48k tokens, for models with a 64k context window. Models with
  a smaller window skip it.

These are in the `full` suite only, so `genbench run --suite standard` skips them. Unless noted, the
stretch and long-context tests need every expectation met to pass (`"pass": {"accuracy": 100}`).

`code-reasoning` tests reading code instead of writing it. Both count toward the Coding column,
every answer is exact, and nothing the model writes is run:

- `code_trace_python`: predict what eight print statements in a 77-line script output. Each turns on a
  different language rule (shared default arguments, late-binding closures, `return` in `finally`,
  round-half-to-even and so on). Seven of eight passes.
- `code_review_planted_bugs`: a 58-line module with numbered lines; five lines don't do what their
  docstring says and three look wrong but are correct. The answer lists the defects by line number.
  (Version 1.1.0 makes the docstring on line 21 promise a new plan per call; in 1.0.0 the shared default
  list on line 20 broke no stated promise, and all five models rightly left it out.)

The expected answers of both are checked by really running the code in `tests/test_code_reasoning_pack.py`.

**Generated context.** A test can generate its own long input instead of storing it:

```json
"context": {"generator": "gateway_logs", "approx_tokens": 16000, "seed": 1601,
            "plants": [{"at": 0.45, "line": "{time} ERROR loader model=... giving up"}],
            "reserved": ["out of memory"]}
```

The text replaces `{{context}}` in the prompt. The same seed always gives the same log, `plants` go at a
fraction of the way through it, and every `reserved` string is checked to appear only in planted lines.
Sizes are estimated without a tokenizer (each digit, word and symbol counts as one token), which is close
for Qwen and Gemma. The Prompt Tokens column shows what the server really counted, and when a prompt
uses 95% or more of the model's context window, the run warns and notes it in the results, because the
server may have cut the prompt. `genbench explain <test> --full-prompt` prints the generated prompt.

Every test explains itself in plain English:

- `name`: the test's name in reports
- `description`: what the model is asked to do
- `goal`: what a passing answer looks like
- `pass`: the score needed to pass, e.g. `{"accuracy": 70}`; omit it to use `[scoring].pass_mark`, or set
  `null` for tests that are only measured
- `expect` on every top-level check: the expectation reports show as met or missed

```json
{
  "expect": "Shows the results in a table",
  "check": "regex",
  "pattern": "st\\.dataframe|st\\.table",
  "points": 10
}
```

Checks available: `contains`, `not_contains`, `regex`, `not_regex`, `all_of`, `any_of`, `none_of`,
`passed`, `word_count`, `code_ratio`, `detect`, `json_valid`, `json_schema`, `yaml_valid`, `tool_calls`,
`numeric_answer`, `choice_answer`, `constraint`, `validator`, `validator_findings`, `external`,
`scaled_score`, `no_tool_calls` (calls that must not be made), `table_row` / `no_table_row` (some row,
or no row, of a markdown table matches all the given patterns) and `table_rows` (row count, and patterns
every row must match). `genbench explain python_streamlit_job_parser` shows a complete example.

Add `"$schema": "../../schemas/testpack.schema.json"` at the top of a pack and VS Code gives you
autocomplete and inline errors.

## Development

```powershell
uv run pytest
```
