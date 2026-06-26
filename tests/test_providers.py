import pytest

from lmeval.providers import get_provider, parse_model_id
from lmeval.providers.anthropic import AnthropicProvider
from lmeval.providers.ollama import OllamaProvider


def test_parse_prefixed_id():
    assert parse_model_id("openai:gpt-4o-mini") == ("openai", "gpt-4o-mini")


def test_parse_keeps_model_tag_colon():
    # provider split happens once; the model's own tag colon is preserved
    assert parse_model_id("ollama:llama3.1:8b") == ("ollama", "llama3.1:8b")


def test_parse_bare_tag_colon_uses_default_provider():
    # "llama3.1" is not a registered provider, so this colon is a tag colon
    assert parse_model_id("llama3.1:8b") == ("ollama", "llama3.1:8b")


def test_parse_bare_id_uses_given_default():
    assert parse_model_id("gpt-4o", default_provider="openai") == ("openai", "gpt-4o")


def test_get_provider_returns_registered_instance():
    assert isinstance(get_provider("ollama"), OllamaProvider)
    assert isinstance(get_provider("anthropic"), AnthropicProvider)


def test_get_provider_is_cached():
    assert get_provider("ollama") is get_provider("ollama")


def test_get_provider_unknown_raises():
    with pytest.raises(ValueError):
        get_provider("nope")
