from lmeval.graders import is_deterministic, run_grader


def test_contains_any_ignorecase():
    r = run_grader({"type": "contains", "any_of": ["positive"], "ignorecase": True}, "Positive.")
    assert r.passed is True


def test_contains_all_of_fails_when_missing():
    r = run_grader({"type": "contains", "all_of": ["alpha", "omega"]}, "only alpha here")
    assert r.passed is False


def test_regex_match():
    r = run_grader({"type": "regex", "pattern": r"\d{3}-\d{4}"}, "call 555-0100 today")
    assert r.passed is True


def test_exact_mismatch():
    r = run_grader({"type": "exact", "value": "yes"}, "no")
    assert r.passed is False


def test_json_schema_valid():
    spec = {"type": "json_schema",
            "schema": {"type": "object", "required": ["x"],
                       "properties": {"x": {"type": "number"}}}}
    assert run_grader(spec, '{"x": 5}').passed is True


def test_json_schema_invalid_json():
    spec = {"type": "json_schema", "schema": {"type": "object", "required": ["x"]}}
    assert run_grader(spec, "not json at all").passed is False


def test_json_extracted_from_fence():
    spec = {"type": "json_schema",
            "schema": {"type": "object", "required": ["a"]}}
    assert run_grader(spec, "```json\n{\"a\": 1}\n```").passed is True


def test_llm_judge_parses_and_passes():
    r = run_grader(
        {"type": "llm_judge", "rubric": "x", "pass_threshold": 4},
        "anything",
        judge_fn=lambda p: '{"score": 5, "rationale": "good"}',
    )
    assert r.passed is True and r.score == 5.0


def test_llm_judge_unparseable_is_none():
    r = run_grader({"type": "llm_judge", "rubric": "x"}, "out",
                   judge_fn=lambda p: "I think it's pretty good!")
    assert r.passed is None


def test_llm_judge_ensemble_averages_scores():
    r = run_grader(
        {"type": "llm_judge", "rubric": "x", "pass_threshold": 4}, "out",
        judge_fns=[lambda p: '{"score": 5, "rationale": "a"}',
                   lambda p: '{"score": 3, "rationale": "b"}'],
    )
    assert r.score == 4.0      # mean of 5 and 3
    assert r.passed is True    # 4.0 >= 4


def test_llm_judge_ensemble_below_threshold_fails():
    r = run_grader(
        {"type": "llm_judge", "rubric": "x", "pass_threshold": 4}, "out",
        judge_fns=[lambda p: '{"score": 5}', lambda p: '{"score": 2}'],
    )
    assert r.score == 3.5
    assert r.passed is False


def test_llm_judge_ensemble_skips_unparseable_members():
    r = run_grader(
        {"type": "llm_judge", "rubric": "x", "pass_threshold": 4}, "out",
        judge_fns=[lambda p: "garbage", lambda p: '{"score": 5}'],
    )
    assert r.score == 5.0      # only the parseable judge counts
    assert r.passed is True


def test_is_deterministic():
    assert is_deterministic({"type": "contains"}) is True
    assert is_deterministic({"type": "llm_judge"}) is False
