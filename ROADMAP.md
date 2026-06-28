# Roadmap

Planned work for llm-eval-harness, roughly in priority order, with the known
limitations that motivate it.

## Known limitations

- **Partial `seed` support.** Only the OpenAI adapter forwards `seed`, so
  cross-provider reproducibility is best-effort.
- **Judge token counts aren't broken out.** Judge calls are priced into
  `judge_cost_usd`, but their tokens aren't added to the token columns (those
  stay the task model's).

## Near term

- An Amazon Bedrock adapter (needs AWS SigV4 signing, so a heavier add than the
  REST/JSON providers — likely an optional `boto3` dependency).

## Medium term

- Cache identical (model, prompt) completions within a run to avoid paying for
  duplicates.
- Per-tag/category score breakdowns in the report.
