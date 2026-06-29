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

---

## Review-driven hardening — positioning, taste, and proof (added 2026-06-28)

> Added after an external code review (captured in `../ai-docs/project_eval/`). This is
> the most *finished and runnable* project of the family — its gap is **category** (a
> well-built instance of a known tool), not completeness. The landscape moved (promptfoo
> is now part of OpenAI; Inspect AI is the open Python standard; Braintrust owns SaaS
> tracing), and the name collides with EleutherAI's `lm-evaluation-harness`. These items
> raise it from "a solid harness" to "this person understands evaluation."
>
> **Definition of Done — the "Sparkle Bar":** a real captured artifact at the top of the
> README · a flagship demo in one screen · stress-tested core · honest numbers with CIs ·
> cold-clone reproducible (`make demo` + CI) · polished surface · one positioning paragraph.

**Hardening items:**
- [ ] **H1 — Position + rename.** Add a "**why this vs promptfoo / Inspect AI / Braintrust / EleutherAI `lm-evaluation-harness`**" section; the defensible lane is *the compact, dependency-light, self-hostable, vendor-neutral, provider-agnostic (incl. local Ollama) Python harness you can read end-to-end and gate CI on — no SaaS, no Node, no account*. **Rename the package** to avoid the `lm-evaluation-harness` collision. *Accept:* README opens with positioning; package renamed.
- [ ] **H2 — Show eval *taste*: ship hard suites.** A **faithfulness / RAG-grounding** suite and a **tool-use / agent** suite with thoughtfully designed rubrics, plus a small **red-team / jailbreak** suite (where the category's energy is). The *suites* should demonstrate judgment about *what's worth measuring*, not just that grading works. *Accept:* the three suites ship and run.
- [ ] **H3 — Upgrade the judge.** Add **pairwise / preference** judging with a **position-swap** to control order bias; document why CI still gates only on deterministic graders. *Accept:* pairwise mode available + tested.
- [ ] **H4 — The real comparison artifact.** Run **3 models on a non-trivial suite**; put the HTML dashboard screenshot at the top of the README. *Accept:* README leads with the real comparison.
- [ ] **H5 — Per-tag / category breakdown** (already listed under *Near term*) — keep; it's the first thing a 200-case suite needs.

*(The "Known limitations" above — partial `seed` forwarding, judge token columns, no live-Bedrock run — remain valid and tracked; folding a live-AWS Bedrock smoke test into CI would close the last one.)*
