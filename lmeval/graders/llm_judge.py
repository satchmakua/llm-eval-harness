"""LLM-as-judge grader: a second model scores the output against a rubric.

This captures subjective qualities (faithfulness, tone, in-character-ness)
that exact-match graders cannot. It is non-deterministic, so it is excluded
from CI gating by default (see --deterministic-only).
"""

import json
import re

from ..types import GradeResult


def grade_llm_judge(output, spec, judge_fn=None, judge_fns=None, **kwargs):
    """Score `output` with one or more judge models; pass on the mean score.

    A single `judge_fn` keeps the original behavior. Pass `judge_fns` (a list)
    to ensemble across several judges -- each scores independently and the
    verdict is the mean vs. `pass_threshold`, which reduces single-model bias.
    """
    fns = list(judge_fns) if judge_fns else ([judge_fn] if judge_fn else [])
    if not fns:
        return GradeResult("llm_judge", passed=None, detail="no judge provider configured")
    prompt = (
        "You are grading an AI response. Grade strictly and fairly.\n"
        f"RUBRIC:\n{spec['rubric']}\n\n"
        f"RESPONSE TO GRADE:\n{output}\n\n"
        'Reply with ONLY JSON: {"score": <integer 1-5>, "rationale": "<one sentence>"}.'
    )
    scores, rationales = [], []
    for fn in fns:
        parsed = _safe_json(fn(prompt))
        if parsed and "score" in parsed:
            scores.append(float(parsed["score"]))
            rationales.append(parsed.get("rationale", ""))
    if not scores:
        return GradeResult("llm_judge", passed=None, score=None,
                           detail=f"no parseable score from {len(fns)} judge(s)")
    mean_score = sum(scores) / len(scores)
    threshold = spec.get("pass_threshold", 4)
    detail = (rationales[0] if len(scores) == 1
              else f"mean {round(mean_score, 2)} across {len(scores)} judges: {scores}")
    return GradeResult("llm_judge", passed=mean_score >= threshold,
                       score=round(mean_score, 2), detail=detail)


def _safe_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None
