# Roadmap

Planned work for llm-eval-harness, roughly in priority order, with the known
limitations that motivate it.

## Known limitations

- **Single-sample judging.** The LLM judge scores once with one model; score
  variance is not measured, and small-N pass rates carry no confidence interval.
- **Partial `seed` support.** Only the OpenAI adapter forwards `seed`, so
  cross-provider reproducibility is best-effort.

## Near term

- Judge ensembling / repeated sampling to quantify score variance.

## Medium term

- Additional providers (e.g. Google Gemini, Amazon Bedrock).
- An HTML dashboard over the JSON results.
