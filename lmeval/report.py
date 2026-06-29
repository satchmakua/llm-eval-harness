"""Aggregate results into per-(suite, model) summaries and write reports."""

import csv
import html
import json
import math
import statistics
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

SUMMARY_COLS = ["suite", "model", "tasks", "passed", "pass_rate",
                "pass_rate_lo", "pass_rate_hi", "mean_judge",
                "prompt_tokens", "completion_tokens", "cost_usd", "judge_cost_usd",
                "p50_latency_s", "p95_latency_s"]


def summarize(results):
    groups = {}
    for r in results:
        groups.setdefault((r.suite, r.model), []).append(r)

    rows = []
    for (suite, model), rs in sorted(groups.items()):
        verdicts = [x.verdict for x in rs if x.verdict is not None]
        passed = sum(1 for v in verdicts if v)
        pass_lo, pass_hi = _wilson(passed, len(verdicts))
        judge_scores = [g.score for x in rs for g in x.grades
                        if g.grader == "llm_judge" and g.score is not None]
        latencies = [x.latency_s for x in rs if x.latency_s]
        rows.append({
            "suite": suite,
            "model": model,
            "tasks": len(rs),
            "passed": passed,
            "pass_rate": round(passed / len(verdicts), 4) if verdicts else None,
            "pass_rate_lo": pass_lo,
            "pass_rate_hi": pass_hi,
            "mean_judge": round(statistics.mean(judge_scores), 2) if judge_scores else None,
            "prompt_tokens": sum(x.prompt_tokens for x in rs),
            "completion_tokens": sum(x.completion_tokens for x in rs),
            "cost_usd": round(sum(x.cost_usd for x in rs), 6),
            "judge_cost_usd": round(sum(x.judge_cost_usd for x in rs), 6),
            "p50_latency_s": round(statistics.median(latencies), 3) if latencies else None,
            "p95_latency_s": round(_percentile(latencies, 95), 3) if latencies else None,
        })
    return rows


def _percentile(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    if lo + 1 < len(s):
        return s[lo] + (s[lo + 1] - s[lo]) * (k - lo)
    return s[lo]


def _wilson(successes, n, z=1.96):
    """Wilson score interval for a binomial proportion (95% by default).

    More honest than the normal approximation at small n and extreme rates --
    e.g. 1/1 gives roughly (0.21, 1.0), a useful "don't trust this yet" signal.
    Returns (low, high) rounded, or (None, None) when there are no trials.
    """
    if n == 0:
        return None, None
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)


def _md_table(rows):
    if not rows:
        return "_no results_"
    header = "| " + " | ".join(SUMMARY_COLS) + " |"
    sep = "| " + " | ".join("---" for _ in SUMMARY_COLS) + " |"
    body = ["| " + " | ".join(str(r.get(c, "")) for c in SUMMARY_COLS) + " |" for r in rows]
    return "\n".join([header, sep, *body])


def _preview(text, limit=200):
    """A single-line, length-bounded preview of (possibly multi-line) text."""
    s = " ".join(str(text).split())
    return s if len(s) <= limit else s[:limit] + "..."


def _failures_md(results, max_items=50):
    """Per-task detail for everything that didn't pass -- the actionable part."""
    failed = [r for r in results if r.verdict is False]
    if not failed:
        return "All graded tasks passed."
    blocks = []
    for r in failed[:max_items]:
        lines = [f"### {r.suite} :: {r.task_id} :: {r.model}"]
        if r.samples > 1 and r.pass_fraction is not None:
            lines.append(f"- samples: {r.samples}, pass fraction: {r.pass_fraction}")
        if r.prompt:
            lines.append(f"- prompt: {_preview(r.prompt)}")
        if r.error:
            lines.append(f"- error: {_preview(r.error)}")
        else:
            lines.append(f"- output: {_preview(r.output)}")
            lines += [f"- failed `{g.grader}`: {g.detail}"
                      for g in r.grades if g.passed is False]
        blocks.append("\n".join(lines))
    hidden = len(failed) - max_items
    if hidden > 0:
        blocks.append(f"_... and {hidden} more failing tasks (see transcripts)._")
    return "\n\n".join(blocks)


def _flaky_md(results):
    """Tasks whose pass/fail flipped across repeated samples (with --repeat > 1)."""
    flaky = [r for r in results
             if r.samples > 1 and r.pass_fraction is not None
             and 0.0 < r.pass_fraction < 1.0]
    return "\n".join(
        f"- {r.suite} :: {r.task_id} :: {r.model} — "
        f"pass fraction {r.pass_fraction} across {r.samples} runs"
        for r in flaky
    )


# ---- HTML dashboard ---------------------------------------------------------
# A self-contained page (inline CSS/JS, no dependencies) for browsing a run.
# Everything model-derived is escaped -- outputs routinely contain markup.

_CSS = """
:root { --pass:#1a7f37; --fail:#cf222e; --na:#6e7781; --line:#d0d7de; --muted:#57606a; }
* { box-sizing: border-box; }
body { font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
       margin: 2rem auto; max-width: 1100px; padding: 0 1rem; color: #1f2328; }
h1 { font-size: 1.4rem; }
h2 { font-size: 1.1rem; margin-top: 2rem; border-bottom: 1px solid var(--line); padding-bottom: .3rem; }
.stamp { color: var(--muted); font-weight: normal; }
table { border-collapse: collapse; width: 100%; font-size: .9rem; }
th, td { text-align: left; padding: .4rem .5rem; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--muted); font-weight: 600; }
.num { font-variant-numeric: tabular-nums; }
.ci { color: var(--muted); font-size: .8rem; }
.badge { display: inline-block; padding: .1rem .4rem; border-radius: .4rem;
         font-size: .75rem; font-weight: 700; color: #fff; }
.badge.pass { background: var(--pass); }
.badge.fail { background: var(--fail); }
.badge.na { background: var(--na); }
td.out { max-width: 440px; white-space: pre-wrap; word-break: break-word; }
.meta { color: var(--muted); font-size: .75rem; margin-top: .25rem; }
ul.grades { margin: 0; padding-left: 1rem; }
ul.grades li.pass { color: var(--pass); }
ul.grades li.fail { color: var(--fail); }
ul.grades li.na { color: var(--na); }
#filter { margin: .5rem 0; padding: .4rem .6rem; width: 320px;
          border: 1px solid var(--line); border-radius: .4rem; }
"""

_JS = """
function _filter() {
  var q = document.getElementById('filter').value.toLowerCase();
  document.querySelectorAll('#tasks tbody tr').forEach(function (r) {
    r.style.display = r.textContent.toLowerCase().indexOf(q) > -1 ? '' : 'none';
  });
}
"""

_VERDICT = {True: ("PASS", "pass"), False: ("FAIL", "fail"), None: ("—", "na")}


def _esc(value):
    return html.escape(str(value), quote=True)


def _fmt(value):
    return "" if value is None else value


def _pass_cell(row):
    pr = row["pass_rate"]
    if pr is None:
        return "—"
    return f"{pr} <span class='ci'>[{row['pass_rate_lo']}–{row['pass_rate_hi']}]</span>"


def _grades_html(grades):
    if not grades:
        return "<span class='na'>—</span>"
    cls = {True: "pass", False: "fail", None: "na"}
    items = []
    for g in grades:
        score = f" ({g.score})" if g.score is not None else ""
        items.append(f"<li class='{cls[g.passed]}'>{_esc(g.grader)}{score}: "
                     f"{_esc(_preview(g.detail, 120))}</li>")
    return "<ul class='grades'>" + "".join(items) + "</ul>"


def _task_row_html(r):
    label, cls = _VERDICT[r.verdict]
    body = _esc(r.error) if r.error else _esc(_preview(r.output, 300))
    meta = ""
    if r.cached:
        meta += "<div class='meta'>served from cache</div>"
    if r.samples > 1 and r.pass_fraction is not None:
        meta += f"<div class='meta'>{r.samples} runs · pass {r.pass_fraction}</div>"
    return (
        f"<tr class='task {cls}'>"
        f"<td><span class='badge {cls}'>{label}</span></td>"
        f"<td>{_esc(r.suite)}</td><td>{_esc(r.task_id)}</td><td>{_esc(r.model)}</td>"
        f"<td class='out'>{body}{meta}</td>"
        f"<td>{_grades_html(r.grades)}</td>"
        "</tr>"
    )


def _html_report(rows, results, stamp):
    summary = "".join(
        "<tr>"
        f"<td>{_esc(r['suite'])}</td><td>{_esc(r['model'])}</td>"
        f"<td class='num'>{r['tasks']}</td><td class='num'>{_pass_cell(r)}</td>"
        f"<td class='num'>{_fmt(r['mean_judge'])}</td>"
        f"<td class='num'>${_fmt(r['cost_usd'])}</td>"
        f"<td class='num'>${_fmt(r['judge_cost_usd'])}</td>"
        f"<td class='num'>{_fmt(r['p50_latency_s'])}</td>"
        f"<td class='num'>{_fmt(r['p95_latency_s'])}</td>"
        "</tr>"
        for r in rows
    ) or "<tr><td colspan='9'>no results</td></tr>"
    tasks = "".join(_task_row_html(r) for r in results)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>lmeval run {_esc(stamp)}</title>
<style>{_CSS}</style>
</head><body>
<h1>Eval run <span class="stamp">{_esc(stamp)}</span></h1>
<h2>Summary</h2>
<table class="summary"><thead><tr>
<th>suite</th><th>model</th><th>tasks</th><th>pass</th><th>judge</th>
<th>cost</th><th>judge cost</th><th>p50 s</th><th>p95 s</th>
</tr></thead><tbody>{summary}</tbody></table>
<h2>Tasks</h2>
<input id="filter" placeholder="filter tasks (suite, id, model, text)…" oninput="_filter()">
<table class="tasks" id="tasks"><thead><tr>
<th>result</th><th>suite</th><th>task</th><th>model</th><th>output / error</th><th>graders</th>
</tr></thead><tbody>{tasks}</tbody></table>
<script>{_JS}</script>
</body></html>
"""


def write_reports(results, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    rows = summarize(results)

    json_path = out / f"run-{stamp}.json"
    json_path.write_text(json.dumps(
        {"summary": rows, "results": [asdict(r) for r in results]}, indent=2))

    csv_path = out / f"summary-{stamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        writer.writeheader()
        writer.writerows(rows)

    md_path = out / f"summary-{stamp}.md"
    sections = [f"# Eval run -- {stamp}", _md_table(rows),
                "## Failures", _failures_md(results)]
    flaky = _flaky_md(results)
    if flaky:
        sections += ["## Flaky tasks", flaky]
    md_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")

    # Per-task transcripts (input, output, grades) as JSONL -- the artifact you
    # grep when a task fails and you need to see exactly what was sent and got back.
    jsonl_path = out / f"transcripts-{stamp}.jsonl"
    with open(jsonl_path, "w", newline="") as f:
        for r in results:
            record = asdict(r)
            record["verdict"] = r.verdict
            f.write(json.dumps(record) + "\n")

    html_path = out / f"dashboard-{stamp}.html"
    html_path.write_text(_html_report(rows, results, stamp), encoding="utf-8")

    return {"json": str(json_path), "csv": str(csv_path), "md": str(md_path),
            "transcripts": str(jsonl_path), "html": str(html_path), "rows": rows}
