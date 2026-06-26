from lmeval.suite import load_suites

SUITE_YAML = """
name: demo
description: a demo suite
models: [ollama:llama3.1:8b]
tasks:
  - id: t1
    system: be terse
    prompt: hello
    graders:
      - type: contains
        any_of: [hi]
"""


def test_load_single_file(tmp_path):
    f = tmp_path / "demo.yaml"
    f.write_text(SUITE_YAML)
    suites = load_suites(f)
    assert len(suites) == 1
    s = suites[0]
    assert s.name == "demo"
    assert s.models == ["ollama:llama3.1:8b"]
    assert len(s.tasks) == 1
    assert s.tasks[0].id == "t1"
    assert s.tasks[0].system == "be terse"


def test_load_directory_globs_yaml_only(tmp_path):
    (tmp_path / "a.yaml").write_text("name: a\ntasks: []\n")
    (tmp_path / "b.yaml").write_text("name: b\ntasks: []\n")
    (tmp_path / "notes.txt").write_text("ignored")
    names = sorted(s.name for s in load_suites(tmp_path))
    assert names == ["a", "b"]


def test_only_filter(tmp_path):
    (tmp_path / "a.yaml").write_text("name: a\ntasks: []\n")
    (tmp_path / "b.yaml").write_text("name: b\ntasks: []\n")
    suites = load_suites(tmp_path, only=["b"])
    assert [s.name for s in suites] == ["b"]


def test_name_defaults_to_filename(tmp_path):
    f = tmp_path / "myfile.yaml"
    f.write_text("tasks: []\n")
    assert load_suites(f)[0].name == "myfile"
