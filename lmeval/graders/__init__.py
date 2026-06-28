"""Grader registry and dispatch."""

from ..types import GradeResult
from .deterministic import DETERMINISTIC
from .llm_judge import grade_llm_judge

_GRADERS = dict(DETERMINISTIC)
_GRADERS["llm_judge"] = grade_llm_judge

DETERMINISTIC_TYPES = set(DETERMINISTIC)


def run_grader(spec, output, judge_fn=None, judge_fns=None):
    fn = _GRADERS.get(spec.get("type"))
    if fn is None:
        return GradeResult(spec.get("type", "?"), passed=None, detail="unknown grader")
    return fn(output, spec, judge_fn=judge_fn, judge_fns=judge_fns)


def is_deterministic(spec):
    return spec.get("type") in DETERMINISTIC_TYPES
