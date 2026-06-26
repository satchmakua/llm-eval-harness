"""Run suites against models and collect scored, costed, timed results."""

from .graders import is_deterministic, run_grader
from .pricing import cost_usd
from .providers import get_provider, parse_model_id
from .types import TaskResult


def _make_judge_fn(judge_model_id, default_provider):
    prov_name, model = parse_model_id(judge_model_id, default_provider)
    provider = get_provider(prov_name)

    def judge_fn(prompt):
        comp = provider.complete(model, [{"role": "user", "content": prompt}],
                                 options={"temperature": 0})
        return comp.text

    return judge_fn


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
        return TaskResult(suite=suite.name, task_id=task.id, model=full_id, error=str(exc))

    grades = []
    for spec in task.graders:
        if deterministic_only and not is_deterministic(spec):
            continue
        judge_fn = None
        if spec.get("type") == "llm_judge":
            judge_fn = _make_judge_fn(spec.get("judge_model", model_id), default_provider)
        grades.append(run_grader(spec, comp.text, judge_fn=judge_fn))

    return TaskResult(
        suite=suite.name,
        task_id=task.id,
        model=full_id,
        output=comp.text,
        prompt_tokens=comp.prompt_tokens,
        completion_tokens=comp.completion_tokens,
        cost_usd=cost_usd(full_id, comp.prompt_tokens, comp.completion_tokens),
        latency_s=comp.latency_s,
        grades=grades,
    )


def run_suites(suites, config, cli_models=None, deterministic_only=False, max_cost=None):
    """Run every (suite, model, task) and collect results.

    `max_cost` is an optional USD budget. It's a soft cap: the run stops before
    starting a task once cumulative spend has reached the budget, so actual
    spend can exceed it by at most the one task that crossed the line. Free
    (local) models never trip it.
    """
    options = config.get("model_options", {})
    default_provider = config.get("default_provider", "ollama")
    results = []
    total_cost = 0.0
    for suite in suites:
        for model_id in resolve_models(suite, config, cli_models):
            for task in suite.tasks:
                if max_cost is not None and total_cost >= max_cost:
                    print(f"  cost budget ${max_cost} reached "
                          f"(spent ${round(total_cost, 6)}); stopping early")
                    return results
                print(f"  {suite.name} :: {model_id} :: {task.id}")
                result = run_task(suite, task, model_id, default_provider,
                                  options, deterministic_only)
                results.append(result)
                total_cost += result.cost_usd
    return results
