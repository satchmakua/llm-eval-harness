# Roadmap

Planned work for llm-eval-harness, roughly in priority order, with the known
limitations that motivate it.

## Known limitations

- **Partial `seed` support.** Only the OpenAI adapter forwards `seed`, so
  cross-provider reproducibility is best-effort.
- **Judge token counts aren't broken out.** Judge calls are priced into
  `judge_cost_usd`, but their tokens aren't added to the token columns (those
  stay the task model's).
- **Bedrock signing isn't exercised against live AWS.** The SigV4 signer is
  verified against the official AWS test vector and the request shape is
  stub-tested, but there are no AWS credentials available here to run a real
  call end to end.

## Near term

- Per-tag/category score breakdowns in the report.

## Medium term

- Persist the completion cache to disk so unchanged tasks are skipped across runs.
