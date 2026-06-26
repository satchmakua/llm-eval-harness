from lmeval.report import summarize
from lmeval.types import GradeResult, TaskResult


def test_summarize_groups_and_counts():
    rows = summarize([
        TaskResult(suite="s", task_id="a", model="openai:gpt-4o",
                   grades=[GradeResult("contains", passed=True)],
                   prompt_tokens=10, completion_tokens=5, cost_usd=0.001, latency_s=1.0),
        TaskResult(suite="s", task_id="b", model="openai:gpt-4o",
                   grades=[GradeResult("contains", passed=False)],
                   prompt_tokens=20, completion_tokens=10, cost_usd=0.002, latency_s=3.0),
    ])
    assert len(rows) == 1
    row = rows[0]
    assert row["tasks"] == 2
    assert row["passed"] == 1
    assert row["pass_rate"] == 0.5
    assert row["prompt_tokens"] == 30
    assert row["completion_tokens"] == 15
    assert row["cost_usd"] == 0.003
    assert row["p50_latency_s"] == 2.0  # median of [1.0, 3.0]


def test_summarize_excludes_undecided_from_pass_rate():
    rows = summarize([
        TaskResult(suite="s", task_id="a", model="m", grades=[]),  # verdict is None
        TaskResult(suite="s", task_id="b", model="m",
                   grades=[GradeResult("contains", passed=True)]),
    ])
    row = rows[0]
    assert row["tasks"] == 2
    assert row["pass_rate"] == 1.0  # only the decided task counts


def test_summarize_mean_judge():
    rows = summarize([
        TaskResult(suite="s", task_id="a", model="m",
                   grades=[GradeResult("llm_judge", passed=True, score=4.0)]),
        TaskResult(suite="s", task_id="b", model="m",
                   grades=[GradeResult("llm_judge", passed=True, score=5.0)]),
    ])
    assert rows[0]["mean_judge"] == 4.5


def test_summarize_splits_by_suite_and_model():
    rows = summarize([
        TaskResult(suite="s1", task_id="a", model="m"),
        TaskResult(suite="s2", task_id="a", model="m"),
    ])
    assert {(r["suite"], r["model"]) for r in rows} == {("s1", "m"), ("s2", "m")}


def test_summarize_empty():
    assert summarize([]) == []
