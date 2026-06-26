from lmeval import runner as runner_mod
from lmeval.runner import resolve_models, run_suites, run_task
from lmeval.types import Completion, Suite, Task


class FakeProvider:
    def __init__(self, text="positive", prompt_tokens=10, completion_tokens=5, raises=None):
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.raises = raises
        self.calls = []

    def complete(self, model, messages, options=None):
        self.calls.append((model, messages, options))
        if self.raises:
            raise self.raises
        return Completion(text=self.text, model=model, provider="fake",
                          prompt_tokens=self.prompt_tokens,
                          completion_tokens=self.completion_tokens, latency_s=0.01)


def _use(monkeypatch, provider):
    monkeypatch.setattr(runner_mod, "get_provider", lambda name, **kw: provider)


def _contains(word):
    return {"type": "contains", "any_of": [word]}


def test_resolve_models_precedence():
    suite = Suite(name="s", tasks=[], models=["suite:m"])
    cfg = {"default_models": ["cfg:m"]}
    assert resolve_models(suite, cfg, ["cli:m"]) == ["cli:m"]      # CLI wins
    assert resolve_models(suite, cfg, None) == ["suite:m"]          # then suite
    assert resolve_models(Suite(name="s", tasks=[]), cfg, None) == ["cfg:m"]  # then config


def test_run_task_success(monkeypatch):
    fake = FakeProvider(text="positive")
    _use(monkeypatch, fake)
    task = Task(id="t", prompt="classify", system="be terse", graders=[_contains("positive")])
    r = run_task(Suite(name="s", tasks=[task]), task, "openai:gpt-4o", "ollama", options={})
    assert r.model == "openai:gpt-4o"
    assert r.output == "positive"
    assert (r.prompt_tokens, r.completion_tokens) == (10, 5)
    assert r.cost_usd > 0
    assert r.verdict is True
    sent_messages = fake.calls[0][1]
    assert sent_messages[0] == {"role": "system", "content": "be terse"}
    assert sent_messages[1] == {"role": "user", "content": "classify"}


def test_run_task_captures_provider_error(monkeypatch):
    _use(monkeypatch, FakeProvider(raises=RuntimeError("boom")))
    task = Task(id="t", prompt="x", graders=[_contains("a")])
    r = run_task(Suite(name="s", tasks=[task]), task, "openai:gpt-4o", "ollama", options={})
    assert r.error == "boom"
    assert r.verdict is False
    assert r.grades == []


def test_deterministic_only_skips_judge(monkeypatch):
    _use(monkeypatch, FakeProvider(text="positive"))
    task = Task(id="t", prompt="x",
                graders=[_contains("positive"), {"type": "llm_judge", "rubric": "r"}])
    r = run_task(Suite(name="s", tasks=[task]), task, "openai:gpt-4o", "ollama",
                 options={}, deterministic_only=True)
    assert [g.grader for g in r.grades] == ["contains"]


def test_run_suites_runs_every_task(monkeypatch):
    _use(monkeypatch, FakeProvider(text="positive"))
    tasks = [Task(id="a", prompt="x", graders=[_contains("positive")]),
             Task(id="b", prompt="y", graders=[_contains("positive")])]
    suite = Suite(name="s", tasks=tasks, models=["openai:gpt-4o"])
    results = run_suites([suite], {"default_provider": "ollama", "model_options": {}})
    assert {r.task_id for r in results} == {"a", "b"}


def test_cost_budget_stops_early(monkeypatch):
    # gpt-4o input is $2.50/1M, so 1M prompt tokens = $2.50 per task.
    _use(monkeypatch, FakeProvider(prompt_tokens=1_000_000, completion_tokens=0))
    tasks = [Task(id=f"t{i}", prompt="x", graders=[_contains("positive")]) for i in range(3)]
    suite = Suite(name="s", tasks=tasks, models=["openai:gpt-4o"])
    results = run_suites([suite], {"default_provider": "ollama", "model_options": {}},
                         max_cost=3.0)
    assert len(results) == 2  # $2.50 + $2.50 = $5.00 >= $3.00 -> stop before the 3rd


def test_cost_budget_ignores_free_models(monkeypatch):
    _use(monkeypatch, FakeProvider(prompt_tokens=1_000_000, completion_tokens=0))
    tasks = [Task(id=f"t{i}", prompt="x", graders=[_contains("positive")]) for i in range(3)]
    suite = Suite(name="s", tasks=tasks, models=["ollama:llama3.1:8b"])  # $0
    results = run_suites([suite], {"default_provider": "ollama", "model_options": {}},
                         max_cost=0.01)
    assert len(results) == 3  # local models cost $0, so the budget never trips
