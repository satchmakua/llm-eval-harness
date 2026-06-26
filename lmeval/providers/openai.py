"""Hosted provider for the OpenAI chat completions API. Needs OPENAI_API_KEY."""

import os
import time

from ..types import Completion
from ._http import post_with_retries
from .base import Provider


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, base_url="https://api.openai.com/v1", timeout=120, max_retries=3):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_key = os.environ.get("OPENAI_API_KEY")

    def available(self):
        return bool(self.api_key)

    def complete(self, model, messages, options=None):
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        body = {"model": model, "messages": messages}
        if options:
            for k in ("temperature", "seed", "max_tokens"):
                if k in options:
                    body[k] = options[k]
        started = time.time()
        resp = post_with_retries(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=body,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        data = resp.json()
        usage = data.get("usage", {})
        return Completion(
            text=data["choices"][0]["message"]["content"],
            model=model,
            provider=self.name,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_s=round(time.time() - started, 3),
        )
