# Roadmap

Planned work for llm-eval-harness, roughly in priority order, with the known
limitations that motivate it.

## Known limitations

- **Partial test coverage.** Graders, gate logic, pricing, reporting, suite
  loading, and the `provider:model` parser are unit-tested; `runner` and `cli`
  are not yet.
- **Sequential execution.** The runner makes one model call at a time, so large
  suites are latency-bound.
- **Single-sample judging.** The LLM judge scores once with one model; score
  variance is not measured, and small-N pass rates carry no confidence interval.
- **Partial `seed` support.** Only the OpenAI adapter forwards `seed`, so
  cross-provider reproducibility is best-effort.

## Near term

- Cover the remaining modules in tests (`runner`, `cli`).
- Optional cost-budget guardrail that aborts a run before it overspends.

## Medium term

- Concurrent task execution.
- Persist raw model outputs for debugging.
- Per-task diffs in the generated reports.

## Later

- Additional providers (e.g. Google Gemini, Amazon Bedrock).
- Judge ensembling / repeated sampling to quantify score variance.
- An HTML dashboard over the JSON results.
