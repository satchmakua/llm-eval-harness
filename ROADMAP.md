# Roadmap

Planned work for llm-eval-harness, roughly in priority order, with the known
limitations that motivate it.

## Known limitations

- **Partial test coverage.** Graders, gate logic, and pricing are unit-tested;
  `runner`, `report`, `suite` loading, the `provider:model` parser, and `cli`
  are not yet.
- **No retry/backoff.** Transient hosted-provider errors (HTTP 429 / 5xx)
  surface as a single errored task instead of being retried.
- **Sequential execution.** The runner makes one model call at a time, so large
  suites are latency-bound.
- **Single-sample judging.** The LLM judge scores once with one model; score
  variance is not measured, and small-N pass rates carry no confidence interval.
- **Partial `seed` support.** Only the OpenAI adapter forwards `seed`, so
  cross-provider reproducibility is best-effort.

## Near term

- Fill the test-coverage gaps above (`report`, `suite`, the id parser, `cli`).
- Retry with exponential backoff in the hosted provider adapters.
- Optional cost-budget guardrail that aborts a run before it overspends.

## Medium term

- Concurrent task execution.
- Persist raw model outputs for debugging.
- Per-task diffs in the generated reports.

## Later

- Additional providers (e.g. Google Gemini, Amazon Bedrock).
- Judge ensembling / repeated sampling to quantify score variance.
- An HTML dashboard over the JSON results.
