# Roadmap

Planned work for llm-eval-harness, roughly in priority order, with the known
limitations that motivate it.

## Known limitations

- **Single judge model.** `--repeat` measures run-to-run variance, but
  ensembling across *different* judge models (to reduce judge bias) isn't
  supported yet.
- **Partial `seed` support.** Only the OpenAI adapter forwards `seed`, so
  cross-provider reproducibility is best-effort.

## Near term

- Additional providers (e.g. Google Gemini, Amazon Bedrock).

## Medium term

- Judge ensembling across multiple judge models.
- An HTML dashboard over the JSON results.
