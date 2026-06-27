"""Token pricing for cost tracking.

Rates are USD per 1,000,000 tokens, as (input, output). The figures below were
taken from each provider's published pricing and verified on 2026-06-27:

  - OpenAI:    https://openai.com/api/pricing/
  - Anthropic: https://platform.claude.com/docs/en/about-claude/models/overview
  - Google:    https://ai.google.dev/gemini-api/docs/pricing

Provider pricing changes over time -- re-check these against the pages above
periodically. Any model not listed here is priced at $0 by cost_usd() (as are
all local Ollama models), so add a row before relying on the dollar figures for
a model you actually run.
"""

PRICING = {
    # OpenAI -- https://openai.com/api/pricing/
    "openai:gpt-4o": (2.50, 10.00),
    "openai:gpt-4o-mini": (0.15, 0.60),
    # Anthropic -- https://platform.claude.com/docs/en/about-claude/models/overview
    "anthropic:claude-opus-4-8": (5.00, 25.00),
    "anthropic:claude-sonnet-4-6": (3.00, 15.00),
    "anthropic:claude-haiku-4-5": (1.00, 5.00),
    # Google Gemini -- https://ai.google.dev/gemini-api/docs/pricing
    # (gemini-2.5-pro rate is the <=200k-token context tier; it rises above that)
    "gemini:gemini-2.5-pro": (1.25, 10.00),
    "gemini:gemini-2.5-flash": (0.30, 2.50),
    "gemini:gemini-2.5-flash-lite": (0.10, 0.40),
}


def cost_usd(model_id, prompt_tokens, completion_tokens):
    """Cost of one call given a normalized 'provider:model' id."""
    if model_id.startswith("ollama:") or ":" not in model_id:
        return 0.0  # local model => free
    rates = PRICING.get(model_id)
    if not rates:
        return 0.0  # unknown model => can't price; report 0 rather than guess
    per_in, per_out = rates
    return round(prompt_tokens / 1e6 * per_in + completion_tokens / 1e6 * per_out, 6)
