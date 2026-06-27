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

- An Amazon Bedrock adapter (needs AWS SigV4 signing, so a heavier add than the
  REST/JSON providers — likely an optional `boto3` dependency).

## Medium term

- Judge ensembling across multiple judge models.
- An HTML dashboard over the JSON results.
