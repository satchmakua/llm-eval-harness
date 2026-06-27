"""Core data structures shared across the harness."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Completion:
    """One model response plus the bookkeeping needed for cost/latency tracking."""
    text: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


@dataclass
class GradeResult:
    """The verdict from a single grader.

    passed: True/False for graders that decide; None for ones that can't
            (e.g. an unparseable judge response).
    score:  raw grader score where applicable (e.g. an LLM judge's 1-5).
    """
    grader: str
    passed: Optional[bool] = None
    score: Optional[float] = None
    detail: str = ""


@dataclass
class Task:
    """A single eval item: a prompt and the graders that judge its output."""
    id: str
    prompt: str
    system: Optional[str] = None
    graders: list = field(default_factory=list)
    expected: Any = None


@dataclass
class Suite:
    """A named group of related tasks."""
    name: str
    description: str = ""
    tasks: list = field(default_factory=list)
    models: Optional[list] = None


@dataclass
class TaskResult:
    """The outcome of running one task against one model.

    For a single run (`samples == 1`) the verdict is derived from the graders.
    With repeated sampling (`samples > 1`), `pass_fraction` holds the share of
    runs that passed and the verdict is a majority vote (strictly > 0.5).
    """
    suite: str
    task_id: str
    model: str
    system: Optional[str] = None
    prompt: str = ""
    output: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    grades: list = field(default_factory=list)
    error: Optional[str] = None
    samples: int = 1
    pass_fraction: Optional[float] = None

    @property
    def verdict(self) -> Optional[bool]:
        """Overall pass/fail.

        Single run: True if every deciding grader passed. Repeated sampling:
        a majority vote over the runs (`pass_fraction` > 0.5; a tie is not a
        pass). None when nothing decided (e.g. manual-review-only), False on a
        hard error.
        """
        if self.error:
            return False
        if self.samples > 1:
            return None if self.pass_fraction is None else self.pass_fraction > 0.5
        decided = [g.passed for g in self.grades if g.passed is not None]
        if not decided:
            return None
        return all(decided)
