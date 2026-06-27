"""Provider registry and the 'provider:model' id parser."""

from .anthropic import AnthropicProvider
from .gemini import GeminiProvider
from .ollama import OllamaProvider
from .openai import OpenAIProvider

_REGISTRY = {
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
}

_cache = {}


def parse_model_id(model_id, default_provider="ollama"):
    """Split a model id into (provider, model).

    'openai:gpt-4o-mini'   -> ('openai', 'gpt-4o-mini')
    'ollama:llama3.1:8b'   -> ('ollama', 'llama3.1:8b')
    'llama3.1:8b'          -> (default_provider, 'llama3.1:8b')   # tag colon, not a provider
    """
    if ":" in model_id:
        head, rest = model_id.split(":", 1)
        if head in _REGISTRY:
            return head, rest
    return default_provider, model_id


def get_provider(name, **kwargs):
    if name not in _REGISTRY:
        raise ValueError(f"unknown provider: {name!r}")
    if name not in _cache:
        _cache[name] = _REGISTRY[name](**kwargs)
    return _cache[name]
