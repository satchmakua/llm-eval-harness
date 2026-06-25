from lmeval.pricing import PRICING, cost_usd


def test_known_model_priced_from_table():
    # gpt-4o is $2.50 in / $10.00 out per 1M -> 1M+1M tokens = $12.50
    assert cost_usd("openai:gpt-4o", 1_000_000, 1_000_000) == 12.5


def test_partial_tokens_match_manual_formula():
    expected = round(1000 / 1e6 * 0.15 + 500 / 1e6 * 0.60, 6)
    assert cost_usd("openai:gpt-4o-mini", 1000, 500) == expected


def test_ollama_models_are_free():
    assert cost_usd("ollama:llama3.1:8b", 9999, 9999) == 0.0


def test_unprefixed_model_is_free():
    # no "provider:" prefix -> nothing to price against -> $0
    assert cost_usd("mystery-model", 1000, 1000) == 0.0


def test_unknown_hosted_model_is_zero_not_error():
    # priced at $0 rather than guessing; never raises
    assert cost_usd("anthropic:claude-does-not-exist", 1000, 1000) == 0.0


def test_pricing_table_entries_are_well_formed():
    for model_id, rates in PRICING.items():
        assert ":" in model_id, f"{model_id} should be a 'provider:model' id"
        assert len(rates) == 2, f"{model_id} needs (input, output) rates"
        per_in, per_out = rates
        assert per_in >= 0 and per_out >= 0, f"{model_id} has a negative rate"
