"""Run suites against models and collect scored, costed, timed results."""

import statistics
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from .graders import is_deterministic, run_grader
from .pricing import cost_usd
from .providers import get_provider, parse_model_id
from .types import TaskResult


def _make_judge_fn(judge_model_id, default_provider, on_cost=None):
    prov_name, model = parse_model_id(judge_model_id, default_provider)
    full_id = f"{prov_name}:{model}"
    provider = get_provider(prov_name)

    def judge_fn(prompt):
        comp = provider.complete(model, [{"role": "user", "content": prompt}],
                                 options={"temperature": 0})
        if on_cost is not None:
            on_cost(cost_usd(full_id, comp.prompt_tokens, comp.completion_tokens))
        return comp.text

    return judge_fn


def _judge_model_ids(spec, fallback):
    """Judge model id(s) for an llm_judge spec: `judge_models` list, a single
    `judge_model`, or the task's own model as a fallback. Always a list."""
    models = spec.get("judge_models") or spec.get("judge_model") or fallback
    return [models] if isinstance(models, str) else list(models)


def resolve_models(suite, config, cli_models):
    """CLI override > suite's own models > config default_models."""
    if cli_models:
        return cli_models
    if suite.models:
        return suite.models
    return config.get("default_models", [])


def run_task(suite, task, model_id, default_provider, options, deterministic_only=False):
    prov_name, model = parse_model_id(model_id, default_provider)
    full_id = f"{prov_name}:{model}"
    provider = get_provider(prov_name)

    messages = []
    if task.system:
        messages.append({"role": "system", "content": task.system})
    messages.append({"role": "user", "content": task.prompt})

    try:
        comp = provider.complete(model, messages, options=options)
    except Exception as exc:
        return TaskResult(suite=suite.name, task_id=task.id, model=full_id,
                          system=task.system, prompt=task.prompt, error=str(exc))

    judge_costs = []  # one entry per judge call, summed into the task's cost
    grades = []
    for spec in task.graders:
        if deterministic_only and not is_deterministic(spec):
            continue
        judge_fns = None
        if spec.get("type") == "llm_judge":
            judge_fns = [_make_judge_fn(jid, default_provider, judge_costs.append)
                         for jid in _judge_model_ids(spec, model_id)]
        grades.append(run_grader(spec, comp.text, judge_fns=judge_fns))

    task_cost = cost_usd(full_id, comp.prompt_tokens, comp.completion_tokens)
    judge_cost = round(sum(judge_costs), 6)
    return TaskResult(
        suite=suite.name,
        task_id=task.id,
        model=full_id,
        system=task.system,
        prompt=task.prompt,
        output=comp.text,
        prompt_tokens=comp.prompt_tokens,
        completion_tokens=comp.completion_tokens,
        cost_usd=round(task_cost + judge_cost, 6),
        judge_cost_usd=judge_cost,
        latency_s=comp.latency_s,
        grades=grades,
    )


def _run_task_sampled(suite, task, model_id, default_provider, options,
                      deterministic_only, repeat):
    """Run one task `repeat` times; collapse to a single majority-vote result.

    `repeat == 1` returns the single run unchanged. Otherwise the runs are
    summed (tokens, cost), averaged (latency), and voted (`pass_fraction`), with
    the first run's input/output kept as a representative sample.
    """
    runs = [run_task(suite, task, model_id, default_provider, options, deterministic_only)
            for _ in range(repeat)]
    if repeat == 1:
        return runs[0]
    decided = [r.verdict for r in runs if r.verdict is not None]
    pass_fraction = round(sum(decided) / len(decided), 4) if decided else None
    first = runs[0]
    return TaskResult(
        suite=first.suite,
        task_id=first.task_id,
        model=first.model,
        system=first.system,
        prompt=first.prompt,
        output=first.output,
        prompt_tokens=sum(r.prompt_tokens for r in runs),
        completion_tokens=sum(r.completion_tokens for r in runs),
        cost_usd=round(sum(r.cost_usd for r in runs), 6),
        judge_cost_usd=round(sum(r.judge_cost_usd for r in runs), 6),
        latency_s=round(statistics.mean([r.latency_s for r in runs]), 3),
        grades=first.grades,
        error=first.error if all(r.error for r in runs) else None,
        samples=repeat,
        pass_fraction=pass_fraction,
    )


def run_suites(suites, config, cli_models=None, deterministic_only=False,
               max_cost=None, workers=1, repeat=1):
    """Run every (suite, model, task) and collect results.

    `workers` runs that many tasks in parallel (default 1 = sequential). Each
    task is one HTTP call, so this is I/O-bound and parallelizes well; results
    are returned in suite/model/task order regardless of completion order.

    `repeat` runs each task that many times and collapses the runs into one
    majority-vote verdict plus a `pass_fraction`, for measuring variance.

    `max_cost` is an optional USD budget. It's a soft cap: no new task is started
    once cumulative spend has reached the budget. With `workers > 1`, up to
    `workers` tasks may already be in flight when the budget trips, so the
    overshoot bound grows with the worker count. Free (local) models never trip
    it.
    """
    options = config.get("model_options", {})
    default_provider = config.get("default_provider", "ollama")
    workers = max(1, workers)

    work = [(suite, model_id, task)
            for suite in suites
            for model_id in resolve_models(suite, config, cli_models)
            for task in suite.tasks]

    results = [None] * len(work)
    total_cost = 0.0
    next_idx = 0
    announced_budget = False

    def over_budget():
        return max_cost is not None and total_cost >= max_cost

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}  # future -> result index

        def submit_ready():
            nonlocal next_idx
            while len(pending) < workers and next_idx < len(work) and not over_budget():
                suite, model_id, task = work[next_idx]
                print(f"  {suite.name} :: {model_id} :: {task.id}")
                future = pool.submit(_run_task_sampled, suite, task, model_id,
                                     default_provider, options, deterministic_only,
                                     repeat)
                pending[future] = next_idx
                next_idx += 1

        submit_ready()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                idx = pending.pop(future)
                result = future.result()
                results[idx] = result
                total_cost += result.cost_usd
            if over_budget() and next_idx < len(work) and not announced_budget:
                print(f"  cost budget ${max_cost} reached "
                      f"(spent ${round(total_cost, 6)}); stopping early")
                announced_budget = True
            submit_ready()  # a no-op once over budget

    return [r for r in results if r is not None]
