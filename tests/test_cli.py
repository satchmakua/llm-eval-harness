from lmeval import cli
from lmeval.types import GradeResult, TaskResult


def _results(passed=True):
    return [TaskResult(suite="classification", task_id="t", model="openai:gpt-4o-mini",
                       grades=[GradeResult("contains", passed=passed)])]


def _suite_dir(tmp_path):
    d = tmp_path / "suites"
    d.mkdir()
    (d / "classification.yaml").write_text(
        "name: classification\n"
        "tasks:\n"
        "  - id: t\n"
        "    prompt: x\n"
        "    graders:\n"
        "      - type: contains\n"
        "        any_of: [x]\n"
    )
    return d


def _common(tmp_path):
    # point --config at a nonexistent file so _load_config returns {}
    return ["--suites", str(_suite_dir(tmp_path)), "--config", str(tmp_path / "none.yaml")]


def test_run_writes_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run_suites", lambda *a, **k: _results())
    out = tmp_path / "out"
    rc = cli.main(["run", *_common(tmp_path), "--out", str(out)])
    assert rc == 0
    assert list(out.glob("run-*.json"))
    assert list(out.glob("summary-*.csv"))


def test_no_suites_returns_1(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run_suites", lambda *a, **k: _results())
    empty = tmp_path / "empty"
    empty.mkdir()
    rc = cli.main(["run", "--suites", str(empty),
                   "--config", str(tmp_path / "none.yaml"), "--out", str(tmp_path / "out")])
    assert rc == 1


def test_run_forwards_max_cost(tmp_path, monkeypatch):
    captured = {}

    def fake_run(suites, config, **kw):
        captured.update(kw)
        return _results()

    monkeypatch.setattr(cli, "run_suites", fake_run)
    cli.main(["run", *_common(tmp_path), "--out", str(tmp_path / "out"), "--max-cost", "1.5"])
    assert captured["max_cost"] == 1.5


def test_baseline_saves_file(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run_suites", lambda *a, **k: _results())
    bdir = tmp_path / "baselines"
    rc = cli.main(["baseline", *_common(tmp_path), "--name", "ci", "--baseline-dir", str(bdir)])
    assert rc == 0
    assert (bdir / "ci.json").exists()


def test_gate_passes_above_floor(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run_suites", lambda *a, **k: _results(passed=True))
    rc = cli.main(["gate", *_common(tmp_path), "--baseline-dir", str(tmp_path / "b"),
                   "--out", str(tmp_path / "out"), "--min-pass-rate", "0.5"])
    assert rc == 0


def test_gate_fails_below_floor(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run_suites", lambda *a, **k: _results(passed=False))
    rc = cli.main(["gate", *_common(tmp_path), "--baseline-dir", str(tmp_path / "b"),
                   "--out", str(tmp_path / "out"), "--min-pass-rate", "0.5"])
    assert rc == 1
