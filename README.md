# genbench

A benchmark for the language models you run on your own hardware. Point it at an Ollama server and it
runs each model through 14 tests (writing code, reading code, design review, agent decisions, and
finding facts in long logs), grades every answer, measures speed, and writes a ranked leaderboard.

- **Nothing is hardcoded.** Settings live in one small `genbench.toml`, and every part of it is optional.
- **Tests are plain JSON.** Each one states what it asks, what a passing answer looks like and how every
  point is earned, and you can add your own without touching the code.
- **Grading is automatic and repeatable.** It never runs the code a model writes.

## What you need

- **[Ollama](https://ollama.com)** with at least one model pulled, on this machine or another one on your
  network. genbench supports Ollama only for now. OpenAI-compatible servers (LM Studio, llama.cpp, vLLM)
  are planned but not built yet.
- **[uv](https://docs.astral.sh/uv/)**, which installs genbench and its own copy of Python. Windows, Linux
  and macOS all work. (Plain `pip` works too; see Install.)

The machine that runs genbench needs no GPU and little memory. It only sends prompts to the server and
grades the text that comes back.

## Quick start

```bash
git clone https://github.com/ybijay/genbench.git genbench
cd genbench
uv sync
uv run genbench run --suite quick -m llama3.2:3b
```

Use any model name that `ollama list` shows in place of `llama3.2:3b`. That runs the four original tests on
one model, which takes a few minutes, and finishes by printing a leaderboard and the folder it wrote the
results to.

From there:

```bash
uv run genbench run                    # all 14 tests; with no config it lists your installed models and asks first
uv run genbench list tests             # see what the tests are
uv run genbench explain code_trace_python    # what one test asks and how it is scored
uv run genbench --help
```

To fix the list of models, put one line in a `genbench.toml` next to the README:

```toml
models = ["llama3.2:3b", "qwen3:8b"]
```

`genbench.example.toml` shows the few other lines people usually need, such as a server on another machine.

## What to expect

- **Time.** The full set of 14 tests took between 17 and 35 minutes per model on a 24 GB GPU with 27B to 35B
  models that think before answering. Small or non-thinking models are much quicker. `--suite quick` runs
  only the four original tests, and `-t TEST` (repeatable) runs the ones you name.
- **Difficulty.** The tests were tuned against 27B to 35B models. Small models will fail many of the
  harder ones; that is the benchmark working, not a setup problem.
- **Context window.** Three tests feed the model long logs (16k, 24k and 48k tokens). A test that doesn't
  fit a model's context window is skipped for that model, and a test any model skipped is left out of
  everyone's scores so the comparison stays fair. With Ollama's small default window, expect those to be
  skipped until you raise it, for example `{ name = "qwen3:8b", options = { num_ctx = 32768 } }`.
- **Theme.** Several tests are set in the world of a self-hosted AI coding platform (an inference gateway,
  a GPU server, a rate limiter). The model needs no background in it; everything is in the prompt.

## Reading the results

For a real example, see [results/2026-11-10_blueboy-p6000.md](results/2026-11-10_blueboy-p6000.md): three
models on a 24 GB NVIDIA Quadro P6000, all 14 tests.

Every run gets its own folder under `runs/`. Open `leaderboard.md` first:

- **Overall** combines accuracy (30%), the Coding column (25%), the Architecture column (20%), generation
  speed (15%) and time to first token (10%). Speed is scored relative to the fastest model in the run, so
  compare Overall only between runs of the same models and tests. The weights are in `[scoring.weights]`.
- **Passed** counts tests that reached their pass mark. Accuracy is whether the answer was right; quality
  is how well it was presented.
- **Results by test** lists, for every test, what was asked, the goal, and exactly which expectations
  each model missed.

If a model's reply was cut off, or it never answered because it filled its context window while thinking,
the report says so in plain words.

## If something goes wrong

| You see | It means | Do this |
| --- | --- | --- |
| `can't reach endpoint 'local'` | Ollama isn't running, or isn't on this machine | Start Ollama, or point genbench at its machine (see Configuration) |
| `model '...' isn't on this server` | The name doesn't match an installed model | Copy the exact name from `ollama list`, or `ollama pull` it |
| `SKIP  Needs a context window of about ...` | The test's prompt is larger than the model's window | Raise `num_ctx` for that model, or ignore it |
| `only 87% of the model fits in VRAM` | Part of the model runs on the CPU, so it will be slow | Use a smaller model or build, or a smaller `num_ctx` |
| `No answer. Ran out of context window` | The model used its whole window thinking | Nothing to fix; that attempt counts as failed |

`uv run genbench validate --online` checks the config, the server and every model name before a long run.

## Install

With [uv](https://docs.astral.sh/uv/) (recommended; it also installs Python if you don't have it):

```bash
uv sync                      # creates .venv with genbench and its dependencies
uv run genbench validate     # run any command through uv
```

To get a `genbench` command everywhere: `uv tool install --editable .`

With pip (needs Python 3.10 or newer):

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e .
genbench validate
```

## Commands

| Command | What it does |
| --- | --- |
| `genbench init [DIR]` | Write a short `genbench.toml` (only what differs from the defaults), plus `packs/` and `schemas/`. `--host` (repeatable), `--import-models models.txt`, `--no-probe`, `--force`; `--full` writes every section |
| `genbench validate` | Check the config and every pack without calling a model. `--online` also contacts each server and checks the models exist; `--portable` warns about regex features the planned Rust version won't support; `--strict` fails on warnings |
| `genbench list tests` | List tests (`--suite`, `--tag`); also `list packs`, `list models`, `list endpoints`, `list runs` |
| `genbench explain TEST` | Show what a test asks, its goal, pass mark and every expectation with its points |
| `genbench new-test` | Start a test from plain-English sentences, then attach a check to each expectation |
| `genbench run` | Run the tests on every enabled model. `--model` and `--test` (repeatable) narrow it down; also `--suite`, `--tag`, `--repeats`. With no models in the config it offers the installed ones (`--all-models` skips the question). Ctrl+C stops cleanly and still writes reports |
| `genbench report RUN` | Rebuild a run folder's reports from its `results.jsonl`, e.g. after changing weights |
| `genbench regrade RUN` | Re-score a run's saved answers with today's packs and config, into a new run folder |
| `genbench regrade --from-python-csv FILE` | Re-score a results CSV saved by the script genbench grew out of, and compare with its scores |
| `genbench clean` | Delete run folders. With no options: only incomplete ones (interrupted, failed, empty, killed). Also `--all`, `--older-than 30d`, `--keep N` (newest completed), or name folders. `--dry-run` previews; it asks before deleting unless `--yes` |
| `genbench schema` | Print the JSON Schema for packs (`--out FILE` to write it) |

Every command takes `--config FILE` and `--set KEY=VALUE` (repeatable) before the command name, e.g.
`genbench --set run.repeats=3 validate`.

## Running a benchmark

```bash
uv run genbench validate --online     # check the config, packs, server and models first
uv run genbench run                   # every enabled model x every test
uv run genbench run -m qwen3:8b -t bash_network_monitor
```

Each run gets its own folder under `runs/` (from `[run].output_dir`), named by date, time and `[run].name`:

| File | Holds |
| --- | --- |
| `results.jsonl` | One line per attempt: timings, the full answer and thinking, every check's result. Everything else is built from it |
| `leaderboard.md` | Ranked table (Overall, Passed, Accuracy, each category, TTFT, tok/s, VRAM), category winners, and for every test what was asked, its goal and each model's missed expectations |
| `results.csv` | One row per attempt for spreadsheets: timings, scores, notes and the answer, then Test ID, Result (PASS, FAIL, SKIPPED or ERROR), Pass Mark, Repeat, Wall TTFT, Error, Quality |
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
[1/5] qwen3.6:35b-a3b-coding-mtp-q4_K_M on local  (MoE)
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

Every test gets two scores from 0 to 100, accuracy (did it do what was asked) and quality (how well),
plus PASS or FAIL against its pass mark. Overall weighs accuracy, the weighted categories and speed by
`[scoring.weights]`.

**Category columns.** Each test adds one number to its category column (Coding, Architecture, Agentic,
Long Context): by default the average of its accuracy and quality, so a wrong answer lowers the column
even when the format was perfect. A test can choose one of the two with `"category_score": "quality"` or
`"accuracy"` in its rubric; the core pack uses `quality`. Harder
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

**Writing the config for you.** `genbench init --host my-gpu-box` (or `--host` several times, or none for
this machine) asks the server which models it has and writes a short `genbench.toml`. Review the file it
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

```bash
uv run pytest
```

The tests need no model server; they run against a small fake one. A few compare genbench with the
script it grew out of, and skip on their own when that script isn't present.

## Background

genbench began as a single Python script with four hardcoded tasks. Those four are now the `core` pack,
and their scoring was kept exactly as it was, which is why that pack scores a little differently from the
others (see Category columns). A Rust version that reads the same config, packs and results is planned.

## License

MIT. See [LICENSE](LICENSE).
