# llm-eval-harness

A small, provider-agnostic harness for **evaluating LLMs the way you'd test any
other system**: define task suites, grade outputs with deterministic checks and
an LLM judge, gate regressions in CI, and track cost and latency per model.

> **Why this exists.** Shipping with LLMs without evals is shipping on vibes.
> An *eval harness* is the testing layer for model-backed systems: it turns
> "does the new prompt/model feel better?" into a number you can diff, gate a
> pull request on, and watch over time. This repo is a compact, readable
> reference implementation of that idea.

## Features

- **Task suites** — eval cases grouped by capability, defined in plain YAML
  (`suites/*.yaml`). No code needed to add a test.
- **Two kinds of graders**
  - *Deterministic*: `exact`, `contains`, `regex`, `json_schema`. Reproducible,
    free, and safe to run in CI.
  - *LLM-as-judge*: a second model scores the output 1-5 against a rubric, for
    subjective qualities (faithfulness, tone) that string matching can't see.
- **Regression gating** — snapshot a baseline, then fail a run (non-zero exit)
  if any suite's pass rate drops below the baseline (with optional tolerance) or
  an absolute floor. Wired into GitHub Actions.
- **Cost & latency tracking** — token counts, USD cost per model (from an
  editable pricing table), and p50/p95 latency, aggregated per suite and model.
- **Provider-agnostic** — one interface, three adapters: **Ollama** (local,
  zero-config, free), **OpenAI**, and **Anthropic**. Models are addressed as
  `provider:model`, e.g. `ollama:llama3.1:8b` or `openai:gpt-4o-mini`.

## How it fits together

```
suites/*.yaml ─► suite loader ─► runner ─► providers (ollama|openai|anthropic)
                                   │
                                   ├─► graders (deterministic + llm-judge)
                                   ├─► pricing (tokens ─► USD)
                                   └─► results ─► report (md / csv / json)
                                                     │
                                                  gate (vs baselines/*.json) ─► exit code
```

## Quickstart (local, no API keys)

```bash
pip install -e ".[dev]"

# requires a local Ollama with the models pulled:
ollama serve
ollama pull llama3.1:8b
ollama pull qwen2.5:7b

lmeval run                       # run every suite against the config defaults
lmeval run --only classification # just one suite
lmeval run --models ollama:qwen2.5:7b
```

Each run prints a summary table (pass rate with a 95% Wilson confidence
interval, mean judge score, cost, and p50/p95 latency) and writes four artifacts
to `results/`: a machine-readable `run-*.json`, a `summary-*.csv` and `summary-*.md`, and a
`transcripts-*.jsonl` with one self-contained record per task (the exact input
sent, the model output, and the grades) — the thing you grep when a task fails.
The `summary-*.md` also carries a **Failures** section listing each failing task
with its prompt/output preview and the specific graders that failed.

## Regression gating

```bash
# 1. record where you stand today
lmeval baseline --name default

# 2. later -- after a prompt edit, model swap, dependency bump -- gate against it
lmeval gate --name default                 # fails (exit 1) on any pass-rate drop
lmeval gate --name default --tolerance 0.1 # allow up to a 10-point drop
lmeval gate --only classification --min-pass-rate 0.8
```

In CI (`.github/workflows/evals.yml`) the `test` job always runs the unit tests.
The `gate` job runs the deterministic suites against a cheap hosted model and
fails the build on a regression — but only if an `OPENAI_API_KEY` repo secret is
configured, so forks and key-less runs skip it cleanly. LLM-judge graders are
left out of CI (`--deterministic-only`) because they aren't reproducible enough
to gate on; run those locally or on a schedule.

## Providers and cost

| provider    | id prefix     | cost        | needs            |
|-------------|---------------|-------------|------------------|
| Ollama      | `ollama:`     | free        | local server     |
| OpenAI      | `openai:`     | per token   | `OPENAI_API_KEY` |
| Anthropic   | `anthropic:`  | per token   | `ANTHROPIC_API_KEY` |

Cost is computed from `lmeval/pricing.py`, which ships current OpenAI and
Anthropic rates (verified 2026-06-09 against each provider's pricing page).
Provider pricing drifts over time, so re-check it periodically. Any model not
listed there — including every local Ollama model — is counted as $0.

Pass `--max-cost <USD>` to any command to cap spend: the run stops before
starting the next task once cumulative cost reaches the budget. It's a soft cap
(actual spend can exceed it by at most the one task that crosses the line, or by
up to `--concurrency` tasks when running in parallel), and free local models
never trip it.

Pass `--concurrency <N>` to run up to N tasks in parallel (default 1). Each task
is a single HTTP call, so this is I/O-bound work that parallelizes well;
results are still reported in a stable suite/model/task order.

Pass `--repeat <N>` to run each task N times; the verdict becomes a majority
vote across the runs, and the report shows each task's pass fraction and flags
any whose result flips. Set a non-zero `temperature` in config for this to
surface real variance — at `temperature: 0` the runs are identical.

## Adding a suite

Drop a YAML file in `suites/`:

```yaml
name: my-suite
description: what this checks
models: [ollama:llama3.1:8b]      # optional; falls back to config defaults
tasks:
  - id: my-task
    system: optional system prompt
    prompt: the user turn
    graders:
      - type: contains
        any_of: ["expected", "word"]
        ignorecase: true
      - type: llm_judge
        judge_model: ollama:llama3.1:8b
        pass_threshold: 4
        rubric: |
          Score 1-5 ...
```

## Grader reference

| type          | passes when …                                              |
|---------------|------------------------------------------------------------|
| `exact`       | output equals `value` (optional `ignorecase`)              |
| `contains`    | output has `any_of` (or every `all_of`) substring          |
| `regex`       | `pattern` matches the output                               |
| `json_schema` | output is valid JSON, optionally matching `schema`         |
| `llm_judge`   | a judge model scores >= `pass_threshold` against `rubric`  |

## Understanding the codebase

**Read the files in this order.**

1. `lmeval/types.py` — the five dataclasses everything else passes around
   (`Task`, `Suite`, `Completion`, `GradeResult`, `TaskResult`). The whole repo
   is functions over these. Note `TaskResult.verdict`: the pass/fail rule.
2. `lmeval/cli.py` — the three subcommands (`run`, `baseline`, `gate`) and how a
   run is wired together. This is the front door.
3. `lmeval/runner.py` — the core loop: every (suite × model × task) becomes one
   `TaskResult`. The single most important file.
4. `lmeval/providers/` — `base.py` is the one-method interface; `ollama.py`,
   `openai.py`, `anthropic.py` are uniform raw-HTTP adapters (hosted ones retry
   transient failures via `_http.py`); `__init__.py` holds the registry and the
   `provider:model` id parser.
5. `lmeval/graders/` — `deterministic.py` (`exact`, `contains`, `regex`,
   `json_schema`) and `llm_judge.py` (a second model scores 1–5 against a rubric).
6. `lmeval/report.py` then `lmeval/gate.py` — aggregation into per-(suite, model)
   summaries, then comparison against a committed baseline.
7. `lmeval/pricing.py` — the token→USD table and the `$0` fallback.

**The one path that matters (a `run`).** Config + `suites/*.yaml` are loaded into
`Suite`/`Task` objects → `runner` iterates suites × models × tasks → for each
task, `parse_model_id` picks the provider, `provider.complete()` returns a
`Completion` (text + token counts + latency), each grader scores the text (an
`llm_judge` grader is handed a `judge_fn` that calls a second model),
`cost_usd` prices the tokens, and it all lands in a `TaskResult`.
`report.summarize()` then groups results by (suite, model) into pass rate, mean
judge score, token/cost sums, and p50/p95 latency; `gate` compares those pass
rates to a baseline and sets the process exit code.

**Concepts worth understanding:**

- **`provider:model` addressing.** Models are `openai:gpt-4o-mini`,
  `ollama:llama3.1:8b`, etc. A bare id uses `default_provider`. The parser is
  careful about the *tag* colon: `llama3.1:8b` splits to provider + `8b` only if
  `llama3.1` is a registered provider (it isn't), so the tag survives intact.
- **Two grader families.** Deterministic graders are reproducible and free, so
  they gate CI. The LLM judge captures subjective quality (faithfulness, tone)
  but isn't reproducible — which is why CI runs `--deterministic-only`.
- **The verdict rule.** A task passes only if *every* deciding grader passes;
  it's `None` when no grader produced a definite verdict, and `False` on error.
  Pass rate is computed over deciding tasks only.
- **Two ways the gate fails.** A relative drop below the baseline (beyond an
  optional `--tolerance`), or falling under an absolute `--min-pass-rate` floor.
  A (suite, model) with no baseline entry is reported but never fails the gate.
- **Cost model.** Local Ollama models and any model not in the pricing table are
  counted as `$0` rather than guessed. `temperature: 0` keeps runs as
  reproducible as the providers allow.

## Roadmap & status

Planned work and known limitations are tracked in [`ROADMAP.md`](ROADMAP.md);
the current implementation status is in [`PROGRESS.md`](PROGRESS.md).

## Layout

```
lmeval/        the package (providers, graders, runner, report, gate, cli)
suites/        eval suites in YAML
baselines/     committed baseline snapshots for gating
tests/         pytest unit tests + a stubbed-HTTP end-to-end test
.github/       CI workflow
```

## License

MIT — see `LICENSE`.
