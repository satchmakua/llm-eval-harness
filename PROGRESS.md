# Progress

Current implementation status of llm-eval-harness. Last updated 2026-06-25.

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
  per-task fault isolation (`lmeval/runner.py`).
- **Reporting** — per-(suite, model) summaries with pass rate, mean judge score,
  token/cost totals, and p50/p95 latency, written as JSON, CSV, and Markdown
  (`lmeval/report.py`).
- **Regression gating** — baseline snapshots plus relative-drop and absolute
  pass-rate floors, with a non-zero exit code on failure; wired into GitHub
  Actions (`lmeval/gate.py`, `.github/workflows/evals.yml`).
- **Cost tracking** — current OpenAI and Anthropic token rates with a `$0`
  fallback for local and unlisted models (`lmeval/pricing.py`).

## Tested

- Deterministic and LLM-judge graders (`tests/test_graders.py`).
- Gate and baseline logic (`tests/test_gate.py`).
- Pricing calculations (`tests/test_pricing.py`).
- Report aggregation (`tests/test_report.py`).
- Suite loading (`tests/test_suite.py`).
- Provider registry and the `provider:model` parser (`tests/test_providers.py`).
- HTTP retry/backoff helper (`tests/test_http_retry.py`).

## Not yet done

Tracked in [`ROADMAP.md`](ROADMAP.md). Highest-priority items: test coverage for
`runner` and `cli`, a cost-budget guardrail, and concurrent execution.
