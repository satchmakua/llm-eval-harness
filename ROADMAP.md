# Roadmap

Planned work for llm-eval-harness, roughly in priority order, with the known
limitations that motivate it.

## Known limitations

- **Judge-call cost is untracked.** Only the task model's tokens are priced;
  LLM-judge calls (including ensembles) are excluded from `cost_usd`.
- **Partial `seed` support.** Only the OpenAI adapter forwards `seed`, so
  cross-provider reproducibility is best-effort.

## Near term

- Track LLM-judge call cost (count judge tokens toward the run's cost).
- An Amazon Bedrock adapter (needs AWS SigV4 signing, so a heavier add than the
  REST/JSON providers — likely an optional `boto3` dependency).

## Medium term

- An HTML dashboard over the JSON results.
