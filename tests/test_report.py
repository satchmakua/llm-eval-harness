import json
from pathlib import Path

from lmeval.report import summarize, write_reports
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


def test_summarize_sums_judge_cost():
    rows = summarize([
        TaskResult(suite="s", task_id="a", model="m", cost_usd=0.003, judge_cost_usd=0.001,
                   grades=[GradeResult("llm_judge", passed=True, score=5.0)]),
        TaskResult(suite="s", task_id="b", model="m", cost_usd=0.002, judge_cost_usd=0.001,
                   grades=[GradeResult("llm_judge", passed=True, score=4.0)]),
    ])
    assert rows[0]["cost_usd"] == 0.005
    assert rows[0]["judge_cost_usd"] == 0.002


def test_summarize_empty():
    assert summarize([]) == []


def test_write_reports_emits_all_artifacts(tmp_path):
    results = [TaskResult(suite="s", task_id="t", model="openai:gpt-4o",
                          system="be terse", prompt="hi", output="positive",
                          grades=[GradeResult("contains", passed=True)])]
    paths = write_reports(results, tmp_path)
    for key in ("json", "csv", "md", "transcripts", "html"):
        assert Path(paths[key]).exists()


def test_transcripts_jsonl_is_self_contained(tmp_path):
    results = [
        TaskResult(suite="s", task_id="t1", model="openai:gpt-4o",
                   system="be terse", prompt="hi", output="positive",
                   grades=[GradeResult("contains", passed=True)]),
        TaskResult(suite="s", task_id="t2", model="openai:gpt-4o",
                   prompt="boom", error="timeout"),
    ]
    paths = write_reports(results, tmp_path)
    lines = Path(paths["transcripts"]).read_text().splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    assert first["prompt"] == "hi"
    assert first["system"] == "be terse"
    assert first["output"] == "positive"
    assert first["verdict"] is True
    assert first["grades"][0]["grader"] == "contains"

    second = json.loads(lines[1])
    assert second["prompt"] == "boom"
    assert second["error"] == "timeout"
    assert second["verdict"] is False  # errored task


def test_md_report_lists_failures_with_detail(tmp_path):
    results = [
        TaskResult(suite="s", task_id="ok", model="m", output="yes",
                   grades=[GradeResult("exact", passed=True, detail="== 'yes'")]),
        TaskResult(suite="s", task_id="bad", model="m", prompt="say yes", output="no",
                   grades=[GradeResult("exact", passed=False, detail="== 'yes'")]),
    ]
    md = Path(write_reports(results, tmp_path)["md"]).read_text(encoding="utf-8")
    assert "## Failures" in md
    assert "s :: bad :: m" in md
    assert "failed `exact`" in md
    assert "say yes" in md          # prompt preview
    assert "output: no" in md
    assert "s :: ok :: m" not in md  # the passing task is not listed


def test_md_report_all_passed(tmp_path):
    results = [TaskResult(suite="s", task_id="ok", model="m",
                          grades=[GradeResult("exact", passed=True)])]
    md = Path(write_reports(results, tmp_path)["md"]).read_text(encoding="utf-8")
    assert "All graded tasks passed." in md


def test_md_report_shows_errored_task(tmp_path):
    results = [TaskResult(suite="s", task_id="boom", model="m", prompt="x", error="timeout")]
    md = Path(write_reports(results, tmp_path)["md"]).read_text(encoding="utf-8")
    assert "error: timeout" in md


def test_pass_rate_ci_is_wide_at_small_n():
    rows = summarize([TaskResult(suite="s", task_id="a", model="m",
                                 grades=[GradeResult("c", passed=True)])])
    row = rows[0]
    assert row["pass_rate"] == 1.0
    # 1/1 is wildly uncertain: the lower bound should be far below 1
    assert 0.0 <= row["pass_rate_lo"] < 0.5
    assert row["pass_rate_hi"] == 1.0
    assert row["pass_rate_lo"] <= row["pass_rate"] <= row["pass_rate_hi"]


def test_pass_rate_ci_tightens_with_more_samples():
    many = [TaskResult(suite="s", task_id=f"t{i}", model="m",
                       grades=[GradeResult("c", passed=True)]) for i in range(50)]
    row = summarize(many)[0]
    assert row["pass_rate"] == 1.0
    assert row["pass_rate_lo"] > 0.9  # 50/50 is far more trustworthy than 1/1


def test_pass_rate_ci_is_none_without_decided_tasks():
    rows = summarize([TaskResult(suite="s", task_id="a", model="m", grades=[])])
    row = rows[0]
    assert row["pass_rate"] is None
    assert row["pass_rate_lo"] is None and row["pass_rate_hi"] is None


def test_md_report_flags_flaky_tasks(tmp_path):
    results = [TaskResult(suite="s", task_id="flaky", model="m", output="x",
                          grades=[GradeResult("c", passed=True)],
                          samples=4, pass_fraction=0.5)]
    md = Path(write_reports(results, tmp_path)["md"]).read_text(encoding="utf-8")
    assert "## Flaky tasks" in md
    assert "pass fraction 0.5 across 4 runs" in md


def test_md_report_no_flaky_section_when_consistent(tmp_path):
    results = [TaskResult(suite="s", task_id="ok", model="m",
                          grades=[GradeResult("c", passed=True)])]
    md = Path(write_reports(results, tmp_path)["md"]).read_text(encoding="utf-8")
    assert "## Flaky tasks" not in md


def test_html_dashboard_is_self_contained_and_escaped(tmp_path):
    results = [
        TaskResult(suite="s", task_id="t1", model="m", output="<script>x</script>",
                   grades=[GradeResult("contains", passed=True)]),
        TaskResult(suite="s", task_id="t2", model="m", prompt="p", error="boom"),
    ]
    page = Path(write_reports(results, tmp_path)["html"]).read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>")
    assert "<style>" in page and "</html>" in page  # inline CSS, complete document
    # model output is escaped, not injected as live markup
    assert "<script>x</script>" not in page
    assert "&lt;script&gt;x&lt;/script&gt;" in page
    # verdict badges and the error both render
    assert "PASS" in page and "FAIL" in page
    assert "boom" in page
