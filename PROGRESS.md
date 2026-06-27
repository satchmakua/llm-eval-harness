# Progress

Current implementation status of llm-eval-harness. Last updated 2026-06-27.

## Implemented

- **Suite loading** — eval suites defined in YAML (`suites/*.yaml`), loaded into
  `Suite`/`Task` objects (`lmeval/suite.py`).
- **Providers** — uniform raw-HTTP adapters for Ollama, OpenAI, and Anthropic
  behind a single interface, addressed as `provider:model`. Hosted adapters
  retry transient failures (HTTP 429 / 5xx, dropped connections) with
  exponential backoff (`lmeval/providers/`).
- **Graders** — deterministic (`exact`, `contains`, `regex`, `json_schema`) and
  an LLM-as-judge grader (`lmeval/graders/`).
- **Runner** — executes every (suite × model × task) into a `TaskResult`, with
  per-task fault isolation, optional parallelism (`--concurrency`, results kept
  in stable order), optional repeated sampling (`--repeat`, majority-vote
  verdict + pass fraction), and an optional `--max-cost` budget that stops a run
  before it overspends (`lmeval/runner.py`).
- **Reporting** — per-(suite, model) summaries with pass rate (and a 95% Wilson
  confidence interval), mean judge score, token/cost totals, and p50/p95 latency
  (JSON, CSV, and a Markdown report that lists each failing task with its output
  and the graders that failed, and flags tasks that flip under `--repeat`),
  plus a `transcripts-*.jsonl` with one self-contained record per task — input
  sent, model output, and grades — for debugging (`lmeval/report.py`).
- **Regression gating** — baseline snapshots plus relative-drop and absolute
  pass-rate floors, with a non-zero exit code on failure; wired into GitHub
  Actions (`lmeval/gate.py`, `.github/workflows/evals.yml`).
- **Cost tracking** — current OpenAI and Anthropic token rates with a `$0`
  fallback for local and unlisted models (`lmeval/pricing.py`).

## Tested

- Deterministic and LLM-judge graders (`tests/test_graders.py`).
- Gate and baseline logic (`tests/test_gate.py`).
- Pricing calculations (`tests/test_pricing.py`).
- Report aggregation, pass-rate confidence intervals, and transcript artifacts
  (`tests/test_report.py`).
- Suite loading (`tests/test_suite.py`).
- Provider registry and the `provider:model` parser (`tests/test_providers.py`).
- HTTP retry/backoff helper (`tests/test_http_retry.py`).
- Runner, including the cost-budget guardrail and repeated-sampling vote
  (`tests/test_runner.py`).
- CLI subcommands and exit codes (`tests/test_cli.py`).
- End-to-end: a real provider adapter over a stub HTTP server, including
  retry/backoff (`tests/test_e2e.py`).

## Not yet done

Tracked in [`ROADMAP.md`](ROADMAP.md). Highest-priority items: additional
providers (Gemini, Bedrock), judge ensembling across multiple judge models, and
an HTML results dashboard.
